import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

import agents as ag
import improve as im

ISBN = '9780306406157'


def toy_snaps():
    cut = ['2023-03-01', '2023-04-01']
    x = pd.DataFrame({'isbn': [ISBN, ISBN], 'cutoff': cut, 'age_months': [5.0, 30.0], 'sale_1m': [1.0, 2.0], 'sale_12m': [12.0, 0.0], 'label': [1, 0]})
    events = pd.DataFrame({'isbn': ISBN, 'date': pd.to_datetime(['2022-03-10', '2023-02-20', '2023-03-15']), 'qty': [5, 3, 4]})
    sales = pd.DataFrame({'isbn': ISBN, 'month': pd.period_range('2022-09', periods=8, freq='M'), 'total': [1, 1, 1, 2, 2, 2, 3, 3]})
    return x, events, sales


class FamiliesTest(unittest.TestCase):
    def test_uses_only_past_records(self):
        x, events, sales = toy_snaps()
        a = im.add_families(x, events, events, sales)
        self.assertEqual(a.ly_event.tolist(), [1, 0])           # 2022-03-10은 2023-03-01의 작년 같은 창
        self.assertEqual(a.event_months_12.tolist()[0], 2)      # 2023-03-15(미래)는 세지 않음
        future = events.copy()
        future.loc[future.date >= '2023-03-01', 'qty'] = 999
        future = pd.concat([future, pd.DataFrame({'isbn': [ISBN], 'date': pd.to_datetime(['2023-03-20']), 'qty': [1]})])
        b = im.add_families(x.iloc[:1], future, future, sales)
        pd.testing.assert_frame_equal(a.iloc[:1], b)
        self.assertEqual(a.semester_start.tolist(), [1, 0])
        self.assertEqual(a.new_6m.tolist(), [1, 0])


class FoldTest(unittest.TestCase):
    def test_folds_roll_forward_and_dev_comes_first(self):
        c = [f'2020-{m:02d}' for m in range(1, 13)] + [f'2021-{m:02d}' for m in range(1, 13)] + [f'2022-{m:02d}' for m in range(1, 13)]
        folds, (dev_fit, dev_eval) = im.make_folds(dict(train=c[:30], val=c[30:], test=[]), 3, 6, 6, min_fit=6)
        self.assertEqual([e[0] for _, e in folds], ['2021-07', '2022-01', '2022-07'])
        for fit, ev in folds:
            self.assertLess(max(fit), min(ev))          # 각 구간은 그 이전 기준일로만 학습
        self.assertLess(max(dev_eval), folds[0][1][0])   # 개발 구간은 첫 검증 구간보다 앞
        with self.assertRaises(ValueError):
            im.make_folds(dict(train=c[:10], val=c[10:16], test=[]), 3, 6, 6)


class SpecTest(unittest.TestCase):
    def test_clamp_limits_and_unknown_family(self):
        s, problems = im.clamp(dict(add_families=['season', 'magic'], max_depth=99, learning_rate=5, train_window_months=3, model='svm'))
        self.assertEqual((s['add_families'], s['max_depth'], s['learning_rate'], s['train_window_months'], s['model']), (['season'], 8, 0.3, 12, 'boosting'))
        self.assertTrue(problems)

    def test_claims_are_checked_against_evidence(self):
        ev = {'R-E1': dict(tool='query_db', input={}, result=dict(columns=['p20'], rows=[[0.5877]]))}
        out = im.verify_claims([dict(text='기준 p20', evidence_id='R-E1', value=58.77), dict(text='지어낸 값', evidence_id='R-E1', value=0.71),
                                dict(text='없는 근거', evidence_id='R-E9', value=0.5), dict(text='열 이름 속 숫자', evidence_id='R-E1', value=20)], ev)
        self.assertEqual([c['verified'] for c in out], [1, 0, 0, 0])


class DbToolTest(unittest.TestCase):
    def test_read_only_select(self):
        path = Path(tempfile.mkdtemp()) / 'p.sqlite'
        db = sqlite3.connect(path); db.executescript(im.SCHEMA); db.execute("INSERT INTO context VALUES('store','교보')"); db.commit(); db.close()
        ws = im.ProcessWS(path)
        self.assertEqual(ws.query_db('select value from context')['rows'], [['교보']])
        for bad in ('delete from context', 'select 1; drop table context', 'select * from final_results'):
            with self.assertRaises(ValueError):
                ws.query_db(bad)
        self.assertNotIn('final_results', ws.describe_db()['tables'])


class FakeClient:
    """역할마다 'DB 조회 1번 → 최종 JSON'. 개선 에이전트는 계절 변수 안, 검토자는 R1-A1을 고르고 수치 주장 2개(맞음·틀림)를 낸다."""

    def __init__(self):
        self.calls = 0
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.calls += 1
        system = kw['system'][0]['text']
        role = next(r for r, t in ag.ROLES.items() if t == system and (r.startswith('improver') or r == 'final_reviewer'))
        last = kw['messages'][-1]['content']
        use = lambda name, inp: NS(type='tool_use', id=f't{self.calls}', name=name, input=inp)
        msg = lambda blocks, stop: NS(content=blocks, stop_reason=stop, model='fake', usage=NS(input_tokens=10, output_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0))
        if isinstance(last, str):
            return msg([use('query_db', {'sql': 'select metric, value from baseline_val'})], 'tool_use')
        pre = ag.PREFIX[role]
        if role == 'final_reviewer':
            out = dict(choice='R1-A1', reasons='계절 변수가 가장 낫다', rejected=[], claims=[
                dict(text='기준 p20', evidence_id='R-E1', value=0.5), dict(text='없는 근거', evidence_id='X-E1', value=1.0)])
        else:
            spec = dict(im.BASE_SPEC, name=f'{role} 계절', add_families=['season'] if role.startswith('improver_A') else [], max_depth=4,
                        rationale='월별 재주문률 차이', evidence_ids=[f'{pre}-E1'])
            spec = {k: spec[k] for k in im.SPEC_SCHEMA['properties']}
            out = dict(critique='' if role.endswith('r1') else '상대 안 평가', proposals=[spec], notes='')
        return msg([NS(type='text', text=json.dumps(out, ensure_ascii=False))], 'end_turn')


class LabStub:
    """작은 가짜 자료로 Lab과 같은 인터페이스를 흉내 낸다."""

    def __init__(self):
        rng = np.random.default_rng(0)
        cuts = pd.date_range('2021-01-01', periods=48, freq='MS').strftime('%Y-%m-%d')
        rows = [dict(isbn=f'{i:013d}', cutoff=c, x1=rng.random(), sale_3m=rng.integers(0, 9), age_months=rng.random() * 60, genre_a=1,
                     label=int(rng.random() < .3)) for c in cuts for i in range(30)]
        self.store, self.base_cols = '교보', ['x1', 'sale_3m', 'age_months']
        self.cuts = dict(train=list(cuts[:36]), val=list(cuts[36:42]), test=list(cuts[42:]))
        self.folds, self.dev = im.make_folds(self.cuts)
        b = pd.DataFrame(rows)
        b['split'] = np.select([b.cutoff.isin(self.cuts['train']), b.cutoff.isin(self.cuts['val'])], ['train', 'val'], 'test')
        b['days_since_rcvd'] = 1.0
        for fam in im.FAMILIES.values():
            for f in fam[1]:
                b[f] = rng.random(len(b))
        self.base = b
        self.books = pd.DataFrame({'isbn': b.isbn.unique(), 'title': 't', 'pub': pd.Timestamp('2020-01-01'), 'price': 1.0, 'genre': 'a', 'status': ''})
        self.tables = {'rcvd': pd.DataFrame({'isbn': [], 'date': pd.to_datetime([]), 'qty': [], 'doc': []}),
                       'rtgd': pd.DataFrame({'isbn': [], 'date': pd.to_datetime([]), 'qty': [], 'doc': []}),
                       'sales': pd.DataFrame({'isbn': [], 'month': pd.PeriodIndex([], freq='M'), 'total': []})}

    snaps = lambda self, cleaning: self.base
    columns = im.Lab.columns
    fit_eval = im.Lab.fit_eval
    evaluate = im.Lab.evaluate

    def recommend(self, s):
        return [dict(rank=1, isbn='0', title='t', score=0.9, cutoff='2023-07-01')]


class ProcessTest(unittest.TestCase):
    def test_full_process_writes_every_stage(self):
        lab = LabStub()
        orig = im.write_stage0
        im.write_stage0 = lambda db, lab: (db.execute("INSERT INTO context VALUES('store','교보')"), db.executemany(
            'INSERT INTO baseline_val VALUES(?,?)', [('p20', 0.5), ('p20_sd', 0.03)]), db.commit(),
            lab.evaluate(im.BASE_SPEC, 'val')[1])[-1]
        try:
            path = Path(tempfile.mkdtemp()) / 'p.sqlite'
            client = FakeClient()
            agents = ag.ApiAgents(client, db=sqlite3.connect(':memory:'))
            db = im.run_store(lab, agents, path, say=lambda m: None)
        finally:
            im.write_stage0 = orig
        count = lambda t: db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
        self.assertEqual(count('proposals'), 6)                 # 3라운드 × 2에이전트 × 1안
        dup = db.execute("SELECT problems FROM proposals WHERE proposal_id='R2-A1'").fetchone()[0]
        self.assertIn('R1-A1와 같은 안', dup)                 # 같은 안은 다시 채점하지 않는다
        self.assertEqual(db.execute('SELECT COUNT(*) FROM rule_check').fetchone()[0], 1)
        self.assertEqual(count('debate'), 6)
        self.assertEqual(db.execute('SELECT choice FROM review').fetchone()[0], 'R1-A1')
        self.assertEqual([r[0] for r in db.execute('SELECT verified FROM claims')], [1, 0])
        self.assertEqual(count('final_results'), 5)
        self.assertEqual(db.execute("SELECT test_delta_vs_base FROM final_results WHERE stage='기준 모델'").fetchone()[0], 0)
        self.assertGreater(count('evidence'), 0)
        self.assertEqual(client.calls, 14)                      # 역할 7개 × (조회 1 + 답 1)


if __name__ == '__main__':
    unittest.main()
