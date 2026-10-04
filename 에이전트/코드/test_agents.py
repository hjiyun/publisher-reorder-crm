import json
import sqlite3
import unittest
from types import SimpleNamespace as NS

import pandas as pd

import agents as ag

ISBN = '9780306406157'


def toy():
    books = pd.DataFrame({'isbn': [ISBN], 'title': ['t'], 'pub': pd.to_datetime(['2021-01-01']), 'price': [15000.0], 'genre': ['문학'], 'status': ['001']})
    dates = pd.to_datetime(['2022-01-03', '2022-01-03', '2022-02-07', '2022-02-07', '2022-03-07', '2022-04-04'])
    rcvd = pd.DataFrame({'isbn': ISBN, 'date': dates, 'qty': [5, 6, 5, 4, 5, 6], 'buy': '위탁', 'center': 'c', 'doc': ['1', '1', '2', '2', '3', '4'], 'rate': 60})
    rtgd = pd.DataFrame({'isbn': ISBN, 'date': pd.to_datetime(['2022-03-10']), 'qty': [1], 'buy': '', 'reason': 'r', 'doc': ['9']})
    sales = pd.DataFrame({'isbn': ISBN, 'month': pd.period_range('2022-01', periods=4, freq='M'), 'store': 1, 'online': 2, 'interpark': 0, 'corp': 0})
    sales['total'] = 3
    return books, {'rcvd': rcvd, 'rtgd': rtgd, 'sales': sales}


def msg(blocks, stop):
    return NS(content=blocks, stop_reason=stop, model='fake', usage=NS(input_tokens=1000, output_tokens=100, cache_read_input_tokens=0, cache_creation_input_tokens=0))


def tool(name, inp, i):
    return NS(type='tool_use', id=f't{i}', name=name, input=inp)


def text(obj):
    return NS(type='text', text=json.dumps(obj, ensure_ascii=False))


ACT = dict(action='drop_key_duplicates', table='sales', keys=['isbn', 'month'], threshold=0, window_months=0, max_days=0, require_multiple_of_10=False, rationale='중복', evidence_ids=[])


class FakeClient:
    """역할마다 '도구 1번 → 최종 JSON' 순서로 답한다. 조정자는 근거 없는 조치 하나를 섞는다."""

    def __init__(self):
        self.calls = 0
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.calls += 1
        system = kw['system'][0]['text']
        last = kw['messages'][-1]['content']
        first = isinstance(last, str)
        prefix = next(p for r, p in ag.PREFIX.items() if ag.ROLES[r] == system)
        if first:
            return msg([tool('key_uniqueness', {'table': 'sales', 'keys': ['isbn', 'month']}, self.calls)], 'tool_use')
        eid = f'{prefix}-E1'
        good = dict(ACT, evidence_ids=[eid])
        if prefix in 'AB':
            return msg([text({'findings': [], 'actions': [good]})], 'end_turn')
        if prefix == 'M':
            return msg([text({'actions': [dict(good, evidence_ids=['A-E1'], support='both'),
                                          dict(ACT, action='drop_nonpositive_qty', table='rcvd', evidence_ids=['X-E9'], support='A')],
                              'dropped': [], 'unresolved': []})], 'end_turn')
        return msg([text({'verdicts': [{'index': 0, 'verdict': 'approve', 'reason': 'ok', 'evidence_ids': [eid], 'modified': good}]})], 'end_turn')


class AgentsTest(unittest.TestCase):
    def test_tools_return_aggregates_without_isbn(self):
        books, t = toy()
        ws = ag.Workspace(books, t)
        out = json.dumps(ag.jsonable([ws.table_overview('rcvd'), ws.key_uniqueness('rcvd', ['ALL']), ws.qty_ratio_outliers('rcvd', 3),
                                      ws.temporal_consistency('rcvd'), ws.doc_date_consistency('rcvd'), ws.referential_integrity('sales'), ws.cross_table_flow()]), ensure_ascii=False)
        self.assertNotIn(ISBN, out)

    def test_actions_keep_index_and_repair(self):
        books, t = toy()
        t['rcvd'].loc[2, ['doc', 'date']] = ['1', pd.Timestamp('2022-01-03')]
        t['rcvd'].loc[1, 'date'] = pd.Timestamp('2023-01-03')  # 문서 1(3행) 안에서 1년 어긋남
        t['rcvd'].loc[4, 'qty'] = 50                            # 단위 오류
        t['sales'] = pd.concat([t['sales'], t['sales'].iloc[[2]]], ignore_index=True)
        out, log = ag.apply_actions(books, t, [dict(action='repair_dates_from_doc', table='rcvd', max_days=180),
                                               dict(action='rescale_qty_outliers', table='rcvd', threshold=5, require_multiple_of_10=True),
                                               dict(action='drop_key_duplicates', table='sales', keys=['isbn', 'month'])])
        self.assertEqual(out['rcvd'].loc[1, 'date'], pd.Timestamp('2022-01-03'))
        self.assertEqual(out['rcvd'].loc[4, 'qty'], 5)
        self.assertEqual(list(out['sales'].index), [0, 1, 2, 3])
        self.assertEqual([x['modified'] for x in log[:2]], [1, 1])

    def test_two_row_doc_tie_is_left_alone(self):
        books, t = toy()
        t['rcvd'].loc[1, 'date'] = pd.Timestamp('2023-01-03')  # 2행 문서에서 어느 쪽이 맞는지 알 수 없다
        out, _ = ag.apply_actions(books, t, [dict(action='repair_dates_from_doc', table='rcvd', max_days=180)])
        pd.testing.assert_series_equal(out['rcvd'].date, t['rcvd'].date)

    def test_preview_does_not_change_data(self):
        books, t = toy()
        ws = ag.Workspace(books, t)
        p = ws.preview_action(dict(action='drop_nonpositive_qty', table='rcvd'))
        self.assertEqual(p['rows_removed'], 0)
        self.assertEqual(len(ws.t['rcvd']), 6)

    def test_pipeline_drops_unsupported_and_caches(self):
        books, t = toy()
        db = sqlite3.connect(':memory:')
        client = FakeClient()
        agents = ag.ApiAgents(client, db=db)
        rv = ag.review(agents, books, t, '테스트')
        self.assertEqual(rv['unsupported_dropped'], 1)
        self.assertEqual([a['action'] for a in rv['actions']], ['drop_key_duplicates'])
        n = client.calls
        ag.review(agents, books, t, '테스트')
        self.assertEqual(client.calls, n)  # 같은 입력은 DB 결과를 다시 쓴다

    def test_schemas_are_strict(self):
        def walk(s):
            if s.get('type') == 'object':
                self.assertFalse(s['additionalProperties'])
                self.assertEqual(set(s['required']), set(s['properties']))
                for v in s['properties'].values():
                    walk(v)
            if s.get('type') == 'array':
                walk(s['items'])
        for schema in [*ag.OUT.values(), *(v[1] for v in ag.TOOLS.values())]:
            walk(schema)


if __name__ == '__main__':
    unittest.main()
