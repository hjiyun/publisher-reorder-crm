"""에이전트 경쟁·토론으로 재주문 예측(매월 다음 30일 재주문 도서 상위 20%)을 개선하는 DB 중심 프로세스.

실행:  python improve.py --store 교보            한 서점
       python improve.py                         교보·예스24
       python improve.py --no-llm                0단계(분석 → DB)와 기준 모델만 (호출 없음)
산출:  ../결과/개선/<run_id>/process_<서점>.sqlite, index.html, run.log, config.json, manifest.json

단계 (모든 입력·중간·최종 산출물을 공정 DB에 저장한다)
0. 분석: 기준 모델의 검증 성능, 월별 성능, 변수 중요도, 오답 구간, 계절별 재주문률, 자료 집계를 DB에 쓴다. 시험 구간 성능은 넣지 않는다.
1. 경쟁: 에이전트 A·B가 DB를 읽고(읽기 전용 SELECT) 서로 모르게 개선안을 낸다. 코드가 검증 구간에서 채점해 DB에 쓴다.
2~3. 토론: A·B가 점수표와 상대 안을 읽고 반박·결합한 수정안을 낸다(수정 최대 2회). 매번 채점해 DB에 쓴다.
4. 적대적 검토: 검토자가 우연한 개선·과도한 시도·근거 없는 주장을 걸러 최종안을 고른다. 수치 주장은 근거 ID의 DB 조회 값과 대조한다.
5. 최종: 단계별 대표안(기준 / 에이전트 1명 / 경쟁 / 토론 / 검토)을 시험 구간에서 한 번씩 평가하고, 최종안으로 다음 달 재주문 예상 도서 목록을 만든다.

에이전트에게 원본 행과 ISBN을 넘기지 않는다. 개선안은 정해진 형식(변수 묶음·모델 설정·학습 기간·정제 조치)으로만 내고 적용은 코드가 한다.
변수 묶음은 모두 기준일 이전 기록만 쓴다(test_improve가 확인).
"""
import argparse
import hashlib
import html
import json
import math
import re
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / '오류실험' / '코드'))

import agents as ag  # noqa: E402
import common  # noqa: E402
import error_injection as ei  # noqa: E402
import kyobo  # noqa: E402

OTHER = {'교보': '예스24', '예스24': '교보'}
FAMILIES = {
    'season': ('기준월의 계절: month_sin·month_cos, 학기 시작월(2·3·8·9월) 여부', ['month_sin', 'month_cos', 'semester_start']),
    'last_year': ('작년 같은 30일 창(기준일 365~335일 전)의 재주문 여부·수량', ['ly_event', 'ly_qty']),
    'reorder_rhythm': ('최근 12개월·3개월 중 재주문이 있었던 달 수', ['event_months_12', 'event_months_3']),
    'other_store': ('다른 서점의 같은 도서 재주문 신호: 최근 30·90일 건수, 작년 같은 창 여부', ['other_n_30d', 'other_n_90d', 'other_ly_event']),
    'sales_momentum': ('판매 추세: 최근 3개월 − 그 전 3개월, 최근 1개월 ÷ 12개월 월평균', ['sale_3m_change', 'sale_1m_vs_avg']),
    'new_release': ('신간 여부: 출간 6개월·12개월 이내', ['new_6m', 'new_12m']),
}
BASE_SPEC = dict(name='기준 모델', add_families=[], drop_features=[], model='boosting', max_depth=3, learning_rate=0.05, max_iter=200,
                 min_samples_leaf=20, class_weight_balanced=False, C=1.0, train_window_months=0, cleaning=[], rationale='현재 파이프라인', evidence_ids=[])
ROUNDS = 3  # 1 경쟁 + 토론 2회
DEV_MONTHS = 6  # 진단표용 개발 구간: 첫 검증 구간 바로 앞의 기준일 수
# 롤링 검증: 시험 구간 직전 N_FOLDS × FOLD_MONTHS개 기준일을 시간 순서대로 나눠, 구간마다 그 이전 기준일로만 학습해 채점한다.
N_FOLDS, FOLD_MONTHS = 3, 6
# 채택 기준(2026-10-06 롤링 검증 도입 시 실행 전에 정함): 세 구간을 합친 짝지은 부트스트랩에서 개선 확률 90% 이상(단측),
# 합친 차이가 양수, 그리고 세 구간 중 2구간 이상에서 기준보다 나을 것.
PROB_RULE, MIN_FOLDS = 0.9, 2
DECISION_RULE = (f'prob_better ≥ {PROB_RULE} 그리고 delta_p20 > 0 그리고 folds_improved ≥ {MIN_FOLDS}/{N_FOLDS} '
                 '(세 검증 구간을 합쳐 같은 도서로 짝지은 부트스트랩 기준, 단측 90%)')


# ---------------------------------------------------------------- 변수 묶음 (기준일 이전 기록만)
def add_families(snaps, events, other, sales):
    x = snaps.reset_index(drop=True).copy()
    c = pd.to_datetime(x.cutoff)
    m = c.dt.month
    x['month_sin'], x['month_cos'] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    x['semester_start'] = m.isin([2, 3, 8, 9]).astype(int)
    groups = x.groupby('cutoff').indices
    cols = {k: np.zeros(len(x)) for k in ('ly_event', 'ly_qty', 'event_months_12', 'event_months_3', 'other_n_30d', 'other_n_90d',
                                          'other_ly_event', 'sale_3m_change')}
    tot = sales.groupby(['isbn', 'month']).total.sum().reset_index()
    for cut, idx in groups.items():
        ct = pd.Timestamp(cut)
        isbn = x.isbn.iloc[idx]
        win = lambda ev, lo, hi: ev[(ev.date >= ct - pd.Timedelta(days=lo)) & (ev.date < ct - pd.Timedelta(days=hi))]
        ly = win(events, 365, 335)
        cols['ly_event'][idx] = isbn.isin(ly.isbn).astype(int).to_numpy()
        cols['ly_qty'][idx] = isbn.map(ly.groupby('isbn').qty.sum()).fillna(0).to_numpy()
        for n, key in ((365, 'event_months_12'), (92, 'event_months_3')):
            w = win(events, n, 0)
            cols[key][idx] = isbn.map(w.groupby('isbn').date.apply(lambda d: d.dt.to_period('M').nunique())).fillna(0).to_numpy()
        if other is not None:
            cols['other_n_30d'][idx] = isbn.map(win(other, 30, 0).groupby('isbn').size()).fillna(0).to_numpy()
            cols['other_n_90d'][idx] = isbn.map(win(other, 90, 0).groupby('isbn').size()).fillna(0).to_numpy()
            cols['other_ly_event'][idx] = isbn.isin(win(other, 365, 335).isbn).astype(int).to_numpy()
        mo = ct.to_period('M')
        last3 = tot[(tot.month >= mo - 3) & (tot.month < mo)].groupby('isbn').total.sum()
        prev3 = tot[(tot.month >= mo - 6) & (tot.month < mo - 3)].groupby('isbn').total.sum()
        cols['sale_3m_change'][idx] = (isbn.map(last3).fillna(0) - isbn.map(prev3).fillna(0)).to_numpy()
    for k, v in cols.items():
        x[k] = v
    x['sale_1m_vs_avg'] = (x.sale_1m / (x.sale_12m / 12)).where(x.sale_12m > 0)
    x['new_6m'], x['new_12m'] = (x.age_months < 6).astype(int), (x.age_months < 12).astype(int)
    return x


# ---------------------------------------------------------------- 개선안 적용·채점
def clamp(spec):
    """개선안의 매개변수를 허용 범위로 맞추고, 쓸 수 없는 항목은 문제로 기록한다."""
    s, problems = dict(BASE_SPEC, **{k: v for k, v in spec.items() if k in BASE_SPEC}), []
    fams = [f for f in s['add_families'] if f in FAMILIES]
    if len(fams) != len(s['add_families']):
        problems.append('알 수 없는 변수 묶음 제외')
    s['add_families'] = sorted(set(fams))
    s['model'] = s['model'] if s['model'] in ('boosting', 'logistic') else 'boosting'
    s['max_depth'] = int(min(max(s['max_depth'] or 3, 2), 8))
    s['learning_rate'] = float(min(max(s['learning_rate'] or 0.05, 0.01), 0.3))
    s['max_iter'] = int(min(max(s['max_iter'] or 200, 50), 600))
    s['min_samples_leaf'] = int(min(max(s['min_samples_leaf'] or 20, 5), 200))
    s['C'] = float(min(max(s['C'] or 1.0, 0.01), 100))
    w = int(s['train_window_months'] or 0)
    s['train_window_months'] = 0 if w <= 0 else min(max(w, 12), 60)
    s['cleaning'] = [{k: a.get(k, ag.ACTION_SCHEMA['properties'][k].get('default')) for k in CLEAN_KEYS} for a in s['cleaning'] if a.get('action') in ag.ACTIONS]
    return s, problems


CLEAN_KEYS = ['action', 'table', 'keys', 'threshold', 'window_months', 'max_days', 'require_multiple_of_10']


def make_model(s):
    if s['model'] == 'logistic':
        return make_pipeline(FunctionTransformer(common.signed_log), SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                             LogisticRegression(C=s['C'], max_iter=3000, class_weight='balanced' if s['class_weight_balanced'] else None))
    return HistGradientBoostingClassifier(max_depth=s['max_depth'], learning_rate=s['learning_rate'], max_iter=s['max_iter'],
                                          min_samples_leaf=s['min_samples_leaf'], class_weight='balanced' if s['class_weight_balanced'] else None,
                                          random_state=0)


def p20_ap(df, scores):
    y = df.label.to_numpy()
    pick = kyobo.monthly_pick(df, scores, common.CONTACT_FRACTION)
    return float(y[pick].mean()), common.metrics(y, scores)['average_precision']


def paired_bootstrap(df, s_new, s_base, reps=300, seed=0):
    """같은 도서를 다시 뽑아(기준일별) 두 점수의 Precision@20% 차이 분포를 만든다. (평균, 표준편차, 개선 확률)"""
    rng = np.random.default_rng(seed)
    groups = list(df.groupby('cutoff').indices.values())
    y = df.label.to_numpy()
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        sub = df.iloc[idx][['cutoff']].reset_index(drop=True)
        a = kyobo.monthly_pick(sub, s_new[idx], common.CONTACT_FRACTION)
        b = kyobo.monthly_pick(sub, s_base[idx], common.CONTACT_FRACTION)
        diffs.append(y[idx][a].mean() - y[idx][b].mean())
    d = np.array(diffs)
    return float(d.mean()), float(d.std(ddof=1)), float((d > 0).mean())


def make_folds(cuts, n_folds=None, size=None, dev=None, min_fit=12):
    """(검증 구간 목록 [(학습 기준일, 평가 기준일)], 개발 구간 (학습 기준일, 평가 기준일)).
    검증 구간은 시험 구간 직전부터 거꾸로 size개씩 n_folds개, 개발 구간은 첫 검증 구간 바로 앞 dev개."""
    n_folds, size, dev = n_folds or N_FOLDS, size or FOLD_MONTHS, dev or DEV_MONTHS
    pre = list(cuts['train']) + list(cuts['val'])
    first = len(pre) - n_folds * size
    if first - dev < min_fit:
        raise ValueError(f'기준일 {len(pre)}개로는 검증 {n_folds}구간·개발 구간을 만들 수 없다')
    folds = [(pre[:first + k * size], pre[first + k * size:first + (k + 1) * size]) for k in range(n_folds)]
    return folds, (pre[:first - dev], pre[first - dev:first])


class Lab:
    """한 서점의 자료·기준일·변수를 들고 개선안을 채점한다. 정제 조치 조합마다 스냅숏을 캐시한다."""

    def __init__(self, store, other_events=None):
        self.store = store
        raw = ei.STORES[store]['load']()[:4]
        _, drop = kyobo.quality(*raw)
        self.books, rcvd, rtgd, sales = kyobo.clean(*raw, drop, True)
        self.tables = {'rcvd': rcvd, 'rtgd': rtgd, 'sales': sales}
        self.cuts = kyobo.cutoffs(rcvd, sales)
        self.folds, self.dev = make_folds(self.cuts)
        self.other = other_events
        self._cache = {}
        self.base = self.snaps([])
        self.base_cols = kyobo.feature_cols(self.base[[c for c in self.base.columns if c not in sum((v[1] for v in FAMILIES.values()), [])]])

    def snaps(self, cleaning):
        key = json.dumps(cleaning, sort_keys=True, ensure_ascii=False)
        if key not in self._cache:
            t = self.tables if not cleaning else ag.apply_actions(self.books, self.tables, cleaning)[0]
            s = kyobo.build(self.books, t['rcvd'], t['rtgd'], t['sales'], self.cuts)
            self._cache[key] = add_families(s, t['rcvd'], self.other, t['sales'])
        return self._cache[key]

    def columns(self, s):
        cols = self.base_cols + [f for fam in s['add_families'] for f in FAMILIES[fam][1]]
        drops = s['drop_features']
        keep = [c for c in cols if not any(c == d or (d.endswith('*') and c.startswith(d[:-1])) for d in drops)]
        return keep or cols

    def fit_eval(self, s, fit_cuts, eval_cuts):
        """fit_cuts로 학습하고 eval_cuts를 채점한다. 평가 도서와 정답은 항상 원자료(기준 스냅숏)로 고정한다.
        정제 조치가 라벨이나 대상을 바꿔도 채점 기준은 그대로이고, 개선안의 자료에서 사라진 평가 도서는 점수 최하위로 둔다."""
        x = self.snaps(s['cleaning'])
        fit = x[x.cutoff.isin(fit_cuts)]
        if s['train_window_months']:
            last = pd.Timestamp(fit.cutoff.max())
            fit = fit[pd.to_datetime(fit.cutoff) > last - pd.DateOffset(months=s['train_window_months'])]
        ev = x[x.cutoff.isin(eval_cuts)]
        cols = self.columns(s)
        target = self.base[self.base.cutoff.isin(eval_cuts)][['isbn', 'cutoff', 'label']].sort_values(['cutoff', 'isbn']).reset_index(drop=True)
        m = target.merge(ev.drop(columns='label'), on=['isbn', 'cutoff'], how='left', indicator=True)
        present = (m.pop('_merge') == 'both').to_numpy()
        sc = np.full(len(m), -1e12)
        if present.any():
            sc[present] = make_model(s).fit(fit[cols], fit.label).predict_proba(m.loc[present, cols])[:, 1]
        return m, sc, cols

    def evaluate(self, s, split='val'):
        """val: 롤링 검증 구간을 모두 합친 결과(fold 열에 구간 번호). test: 시험 직전까지 모두 학습하고 시험 구간 채점."""
        if split == 'test':
            return self.fit_eval(s, list(self.cuts['train']) + list(self.cuts['val']), self.cuts['test'])
        parts = [self.fit_eval(s, f, e) for f, e in self.folds]
        ev = pd.concat([m.assign(fold=k + 1) for k, (m, _, _) in enumerate(parts)], ignore_index=True)
        return ev, np.concatenate([sc for _, sc, _ in parts]), parts[0][2]

    def recommend(self, s, top=0.2):
        """최종안을 라벨이 있는 모든 기준일로 학습하고, 판매 자료 다음 달 1일 기준으로 도서를 고른다."""
        x = self.snaps(s['cleaning'])
        cols = self.columns(s)
        t = self.tables if not s['cleaning'] else ag.apply_actions(self.books, self.tables, s['cleaning'])[0]
        # 마지막 판매 달이 끝나지 않았으면(예: 예스24 10월 1~3일) 그 달 1일을 기준일로 한다
        cut = min((t['sales'].month.max() + 1).to_timestamp(), pd.Timestamp.today().normalize().replace(day=1))
        nxt = kyobo.snapshot(self.books, t['rcvd'], t['rtgd'], t['sales'], cut).assign(cutoff=str(cut.date()), label=0, split='next')
        nxt = add_families(nxt, t['rcvd'], self.other, t['sales'])
        sc = make_model(s).fit(x[cols], x.label).predict_proba(nxt[cols])[:, 1]
        k = max(1, math.ceil(len(nxt) * top))
        order = np.argsort(-sc, kind='stable')[:k]
        titles = self.books.set_index('isbn').title
        return [dict(rank=i + 1, isbn=nxt.isbn.iloc[j], title=str(titles.get(nxt.isbn.iloc[j], '')), score=float(sc[j]), cutoff=str(cut.date()))
                for i, j in enumerate(order)]


# ---------------------------------------------------------------- 공정 DB
SCHEMA = '''
CREATE TABLE context(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE feature_catalog(feature TEXT, family TEXT, in_baseline INTEGER, description TEXT);
CREATE TABLE candidate_families(family TEXT, description TEXT, features TEXT);
CREATE TABLE baseline_val(metric TEXT, value REAL);
CREATE TABLE dev_by_month(cutoff TEXT, books INTEGER, reorders INTEGER, base_rate REAL, picked INTEGER, hits INTEGER, p20 REAL);
CREATE TABLE feature_importance(feature TEXT, ap_drop REAL);
CREATE TABLE error_segments(segment_type TEXT, segment TEXT, books INTEGER, reorders INTEGER, picked INTEGER, hits INTEGER, missed INTEGER);
CREATE TABLE train_month_rate(calendar_month INTEGER, rows INTEGER, reorder_rate REAL);
CREATE TABLE family_signal(family TEXT, feature TEXT, corr_with_label_train REAL);
CREATE TABLE data_profile(tbl TEXT, tool TEXT, result TEXT);
CREATE TABLE proposals(proposal_id TEXT PRIMARY KEY, round INTEGER, agent TEXT, name TEXT, spec TEXT, rationale TEXT, evidence_ids TEXT, problems TEXT);
CREATE TABLE scores(proposal_id TEXT PRIMARY KEY, round INTEGER, agent TEXT, val_p20 REAL, val_ap REAL, delta_p20 REAL, delta_sd REAL, prob_better REAL, delta_ap REAL, n_features INTEGER, fold_deltas TEXT, folds_improved INTEGER);
CREATE TABLE debate(round INTEGER, agent TEXT, critique TEXT, notes TEXT);
CREATE TABLE review(choice TEXT, reasons TEXT, payload TEXT);
CREATE TABLE claims(agent TEXT, text TEXT, evidence_id TEXT, value REAL, verified INTEGER, note TEXT);
CREATE TABLE evidence(evidence_id TEXT PRIMARY KEY, role TEXT, tool TEXT, input TEXT, result TEXT);
CREATE TABLE agent_log(role TEXT, seconds REAL, cost_usd REAL, tool_calls TEXT, cached INTEGER);
CREATE TABLE final_results(stage TEXT, proposal_id TEXT, val_p20 REAL, test_p20 REAL, test_ap REAL, test_delta_vs_base REAL);
CREATE TABLE recommendations(rank INTEGER, isbn TEXT, title TEXT, score REAL, cutoff TEXT);
'''
DB_GUIDE = {
    'context': '서점, 라벨 정의, 기준일 구간, 검증 구간 규모(도서 1권 = 몇 %p) 등',
    'feature_catalog': '기준 모델 변수와 후보 변수(묶음별)',
    'candidate_families': '추가할 수 있는 변수 묶음과 설명',
    'baseline_val': '기준 모델의 검증 구간 성능(p20, ap)과 부트스트랩 표준편차(p20_sd)',
    'dev_by_month': '개발 구간(학습 구간 마지막 6개 기준일) 기준일별 재주문률·적중',
    'feature_importance': '개발 구간에서 변수를 섞었을 때 AP 하락폭(클수록 중요)',
    'error_segments': '개발 구간 오답 분석: 분야·출간 경과·최근 판매·달력 월 구간별 재주문·선정·적중·놓침',
    'train_month_rate': '학습 구간 달력 월별 재주문률(계절성)',
    'family_signal': '후보 변수와 학습 라벨의 상관(학습 구간)',
    'data_profile': '자료 집계(표 개요·키 중복·문서번호 일관성 등, JSON)',
    'proposals': '지금까지의 개선안(spec JSON 포함)',
    'scores': '개선안의 검증 점수(세 검증 구간 합산): val_p20, delta_p20(기준 대비), delta_sd(짝지은 부트스트랩 표준편차), prob_better(개선 확률), fold_deltas(구간별 기준 대비 차이), folds_improved(기준보다 나은 구간 수)',
    'debate': '라운드별 상대 안 평가',
}


def write_stage0(db, lab):
    """0단계: 기준 모델 분석 결과를 집계로만 DB에 쓴다(시험 구간 성능은 넣지 않는다)."""
    s = dict(BASE_SPEC)
    ev, sc, cols = lab.evaluate(s, 'val')
    p20, ap = p20_ap(ev, sc)
    boot = [p20_ap(ev.iloc[i].reset_index(drop=True), sc[i])[0] for i in
            (np.concatenate([np.random.default_rng(k).choice(g, len(g)) for g in ev.groupby('cutoff').indices.values()]) for k in range(200))]
    per_book = 1 / max(1, int(sum(math.ceil(len(g) * common.CONTACT_FRACTION) for g in ev.groupby('cutoff').indices.values())))
    ctx = dict(store=lab.store, event='입하' if lab.store == '교보' else '발주', label='기준일 이후 30일 안에 재주문 1건 이상',
               train_cutoffs=f'{lab.cuts["train"][0]} ~ {lab.cuts["train"][-1]} ({len(lab.cuts["train"])}개)',
               val_folds=' / '.join(f'구간{k + 1} {e[0]} ~ {e[-1]}' for k, (_, e) in enumerate(lab.folds)) + ' (구간마다 그 이전 기준일로만 학습)',
               test='시험 구간은 최종 평가 전까지 공개하지 않음', val_rows=len(ev), val_selected=int(round(1 / per_book)),
               one_book_in_p20=f'{per_book * 100:.2f}%p', other_store=OTHER[lab.store], decision_rule=DECISION_RULE)
    db.executemany('INSERT INTO context VALUES(?,?)', [(k, str(v)) for k, v in ctx.items()])
    for c in lab.base_cols:
        db.execute('INSERT INTO feature_catalog VALUES(?,?,?,?)', (c, 'baseline', 1, ''))
    for fam, (desc, feats) in FAMILIES.items():
        db.execute('INSERT INTO candidate_families VALUES(?,?,?)', (fam, desc, ', '.join(feats)))
        for f in feats:
            db.execute('INSERT INTO feature_catalog VALUES(?,?,?,?)', (f, fam, 0, desc))
    db.executemany('INSERT INTO baseline_val VALUES(?,?)', [('p20', p20), ('ap', ap), ('p20_sd', float(np.std(boot, ddof=1))),
                                                            ('rule_p20', p20_ap(ev, kyobo.rule_score(ev))[0])] +
                   [(f'p20_fold{k}', p20_ap(ev[ev.fold == k].reset_index(drop=True), sc[(ev.fold == k).to_numpy()])[0]) for k in sorted(ev.fold.unique())])
    # 진단표는 첫 검증 구간보다 앞선 개발 구간에서 계산한다. 검증 구간은 채점에만 쓴다(트러블슈팅 #21, #22).
    dev_fit, dev_cuts = lab.dev
    x = lab.base
    fit = x[x.cutoff.isin(dev_fit)]
    dev = x[x.cutoff.isin(dev_cuts)].sort_values(['cutoff', 'isbn']).reset_index(drop=True)
    model = make_model(s).fit(fit[cols], fit.label)
    dsc = model.predict_proba(dev[cols])[:, 1]
    dev_ap = common.metrics(dev.label.to_numpy(), dsc)['average_precision']
    db.executemany('INSERT INTO context VALUES(?,?)', [('dev_cutoffs', f'{dev_cuts[0]} ~ {dev_cuts[-1]} ({len(dev_cuts)}개, 진단표 계산용. 첫 검증 구간보다 앞)'),
                                                      ('dev_p20', f'{p20_ap(dev, dsc)[0]:.4f}')])
    pick = kyobo.monthly_pick(dev, dsc, common.CONTACT_FRACTION)
    dev = dev.assign(picked=pick.astype(int), hit=(pick & (dev.label == 1)).astype(int))
    for cut, g in dev.groupby('cutoff'):
        db.execute('INSERT INTO dev_by_month VALUES(?,?,?,?,?,?,?)', (cut, len(g), int(g.label.sum()), float(g.label.mean()), int(g.picked.sum()),
                                                                     int(g.hit.sum()), float(g.hit.sum() / g.picked.sum())))
    rng = np.random.default_rng(0)
    for c in cols:
        shuffled = dev[cols].copy()
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
        db.execute('INSERT INTO feature_importance VALUES(?,?)', (c, dev_ap - common.metrics(dev.label.to_numpy(), model.predict_proba(shuffled)[:, 1])['average_precision']))
    genre = dev[[c for c in dev.columns if c.startswith('genre_')]].idxmax(axis=1).str.replace('genre_', '')
    segs = {'분야': genre, '출간 경과': pd.cut(dev.age_months, [0, 6, 12, 24, 48, 1e9], right=False, labels=['6개월 미만', '6~12개월', '1~2년', '2~4년', '4년 이상']),
            '최근 3개월 판매': pd.cut(dev.sale_3m, [-1e9, 0.5, 5, 20, 1e9], labels=['0', '1~5', '6~20', '21+']),
            '달력 월': pd.to_datetime(dev.cutoff).dt.month.astype(str) + '월'}
    for name, seg in segs.items():
        for k, g in dev.groupby(seg.astype(str)):
            db.execute('INSERT INTO error_segments VALUES(?,?,?,?,?,?,?)', (name, k, len(g), int(g.label.sum()), int(g.picked.sum()), int(g.hit.sum()),
                                                                           int(((g.label == 1) & (g.picked == 0)).sum())))
    tr = lab.base[lab.base.cutoff.isin(list(dev_fit) + list(dev_cuts))]  # 학습 구간 통계도 검증 구간 이전 자료로만
    for mth, g in tr.groupby(pd.to_datetime(tr.cutoff).dt.month):
        db.execute('INSERT INTO train_month_rate VALUES(?,?,?)', (int(mth), len(g), float(g.label.mean())))
    for fam, (_, feats) in FAMILIES.items():
        for f in feats:
            v = tr[f].astype(float)
            db.execute('INSERT INTO family_signal VALUES(?,?,?)', (fam, f, float(v.corr(tr.label.astype(float))) if v.std() > 0 else None))
    ws = ag.Workspace(lab.books, lab.tables)
    for t in ag.TABLES:
        for tool, args in (('table_overview', {}), ('key_uniqueness', {'keys': ['ALL']}), ('qty_ratio_outliers', {'threshold': 8}), ('temporal_consistency', {})):
            db.execute('INSERT INTO data_profile VALUES(?,?,?)', (t, tool, json.dumps(ag.jsonable(getattr(ws, tool)(t, **args)), ensure_ascii=False)))
        if t != 'sales':
            db.execute('INSERT INTO data_profile VALUES(?,?,?)', (t, 'doc_date_consistency', json.dumps(ag.jsonable(ws.doc_date_consistency(t)), ensure_ascii=False)))
    db.commit()
    return sc


def same_spec(a, b):
    keys = [k for k in BASE_SPEC if k not in ('name', 'rationale', 'evidence_ids')]
    return all(json.dumps(a[k], sort_keys=True) == json.dumps(b[k], sort_keys=True) for k in keys)


def score_proposal(db, lab, base_scores, pid, rnd, agent, spec):
    s, problems = clamp(spec)
    for old_id, old in db.execute('SELECT proposal_id, spec FROM proposals WHERE proposal_id != ?', (pid,)).fetchall():
        if same_spec(s, json.loads(old)):
            db.execute('INSERT OR REPLACE INTO proposals VALUES(?,?,?,?,?,?,?,?)', (pid, rnd, agent, spec.get('name', ''), json.dumps(s, ensure_ascii=False),
                                                                                    spec.get('rationale', ''), json.dumps(spec.get('evidence_ids', [])), f'{old_id}와 같은 안(다시 채점하지 않음)'))
            row = db.execute('SELECT val_p20, val_ap, delta_p20, delta_sd, prob_better, delta_ap, n_features, fold_deltas, folds_improved FROM scores WHERE proposal_id=?', (old_id,)).fetchone()
            db.execute('INSERT OR REPLACE INTO scores VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', (pid, rnd, agent, *row))
            db.commit()
            return s
    db.execute('INSERT OR REPLACE INTO proposals VALUES(?,?,?,?,?,?,?,?)', (pid, rnd, agent, spec.get('name', ''), json.dumps(s, ensure_ascii=False),
                                                                            spec.get('rationale', ''), json.dumps(spec.get('evidence_ids', [])), '; '.join(problems)))
    try:
        ev, sc, cols = lab.evaluate(s, 'val')
        p20, ap = p20_ap(ev, sc)
        d, sd, prob = paired_bootstrap(ev, sc, base_scores)
        base_p20, base_ap = p20_ap(ev, base_scores)
        folds = []
        for k in sorted(ev.fold.unique()):
            mask = (ev.fold == k).to_numpy()
            sub = ev[mask].reset_index(drop=True)
            folds.append(round(p20_ap(sub, sc[mask])[0] - p20_ap(sub, base_scores[mask])[0], 4))
        row = (pid, rnd, agent, p20, ap, p20 - base_p20, sd, prob, ap - base_ap, len(cols), json.dumps(folds), sum(f > 0 for f in folds))
    except Exception as e:
        db.execute('UPDATE proposals SET problems=? WHERE proposal_id=?', ('; '.join(problems + [f'채점 실패: {e}']), pid))
        row = (pid, rnd, agent, None, None, None, None, None, None, None, None, None)
    db.execute('INSERT OR REPLACE INTO scores VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', row)
    db.commit()
    return s


# ---------------------------------------------------------------- 에이전트 (agents.py의 호출 계층을 그대로 쓴다)
class ProcessWS:
    """에이전트가 쓰는 공정 DB 도구. 읽기 전용 연결로 SELECT만 실행한다."""

    def __init__(self, path):
        self.path = path

    def _ro(self):
        con = sqlite3.connect(f'file:{Path(self.path).as_posix()}?mode=ro', uri=True)
        con.execute('PRAGMA query_only=ON')
        return con

    def describe_db(self):
        con = self._ro()
        out = {}
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            if t in ('evidence', 'agent_log', 'final_results', 'recommendations', 'claims', 'review', 'rule_check'):
                continue
            cols = [r[1] for r in con.execute(f'PRAGMA table_info({t})')]
            n = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
            out[t] = dict(columns=cols, rows=n, about=DB_GUIDE.get(t, ''))
        con.close()
        return dict(tables=out)

    def query_db(self, sql):
        q = sql.strip().rstrip(';')
        if ';' in q or not re.match(r'(?is)^\s*(select|with)\b', q):
            raise ValueError('SELECT 문 하나만 실행할 수 있다')
        if re.search(r'(?i)\b(evidence|agent_log|final_results|recommendations)\b', q):
            raise ValueError('이 테이블은 조회할 수 없다')
        con = self._ro()
        cur = con.execute(q)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(101)
        con.close()
        return dict(columns=cols, rows=[list(r) for r in rows[:100]], truncated=len(rows) > 100)


SPEC_SCHEMA = ag._strict({
    'name': {'type': 'string'},
    'add_families': {'type': 'array', 'items': {'type': 'string', 'enum': list(FAMILIES)}},
    'drop_features': {'type': 'array', 'items': {'type': 'string'}},
    'model': {'type': 'string', 'enum': ['boosting', 'logistic']},
    'max_depth': {'type': 'integer'}, 'learning_rate': {'type': 'number'}, 'max_iter': {'type': 'integer'},
    'min_samples_leaf': {'type': 'integer'}, 'class_weight_balanced': {'type': 'boolean'}, 'C': {'type': 'number'},
    'train_window_months': {'type': 'integer'},
    'cleaning': {'type': 'array', 'items': ag._strict({k: ag.ACTION_SCHEMA['properties'][k] for k in CLEAN_KEYS})},
    'rationale': {'type': 'string'},
    'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
})
IMPROVER_OUT = ag._strict({'critique': {'type': 'string'}, 'proposals': {'type': 'array', 'items': SPEC_SCHEMA}, 'notes': {'type': 'string'}})
CLAIM = ag._strict({'text': {'type': 'string'}, 'evidence_id': {'type': 'string'}, 'value': {'type': 'number'}})
FINAL_OUT = ag._strict({'choice': {'type': 'string'}, 'reasons': {'type': 'string'},
                        'rejected': {'type': 'array', 'items': ag._strict({'proposal_id': {'type': 'string'}, 'reason': {'type': 'string'}})},
                        'claims': {'type': 'array', 'items': CLAIM}})

PROCESS_COMMON = '''당신은 출판사의 재주문 예측을 개선하는 데이터 분석 에이전트입니다.
과제: 매월 1일 기준으로, 다음 30일 안에 서점이 재주문할 도서 상위 20%를 고릅니다. 지표는 기준일마다 상위 20%를 고른 뒤 합친 Precision@20%(p20)입니다.
- 공정 DB(SQLite)에 분석 결과가 집계로만 들어 있습니다. describe_db로 테이블을 확인하고 query_db(SELECT 하나)로 조회하세요. 결과마다 근거 ID가 붙습니다.
- 원본 행과 도서 식별자는 볼 수 없고, 시험 구간 성능은 최종 평가 전까지 공개되지 않습니다. 채점은 롤링 검증으로 합니다: 시험 직전 18개 기준일을 6개씩 세 구간으로 나누고, 구간마다 그 이전 기준일로만 학습해 채점한 뒤 합칩니다(context의 val_folds). 진단표(dev_by_month, feature_importance, error_segments, train_month_rate, family_signal)는 첫 검증 구간보다 앞선 자료로만 계산했습니다.
- 개선안(spec)으로 바꿀 수 있는 것:
  · add_families: 추가 변수 묶음(candidate_families 테이블). 모두 기준일 이전 기록만 씁니다.
  · drop_features: 뺄 변수 이름(끝에 *를 붙이면 접두어, 예: genre_*)
  · model(boosting|logistic)과 설정: max_depth 2~8, learning_rate 0.01~0.3, max_iter 50~600, min_samples_leaf 5~200, class_weight_balanced, C 0.01~100(logistic)
  · train_window_months: 0이면 학습 구간 전체, 12~60이면 최근 N개월 기준일만 학습
  · cleaning: 데이터 정제 조치(data_profile 테이블의 집계를 근거로). 정상 기록을 지우면 재주문 라벨이 사라지므로 신중히.
- 쓰지 않는 값은 기준 모델 값(boosting, max_depth 3, learning_rate 0.05, max_iter 200, min_samples_leaf 20, C 1.0, 나머지는 비움·0·false)으로 두세요.
- 검증 구간은 작습니다. 두 모델 비교는 scores의 delta_sd(같은 도서로 짝지어 잰 차이의 표준편차), prob_better(개선 확률), fold_deltas·folds_improved(구간별 일관성)로 판단하세요. baseline_val의 p20_sd는 점수 하나의 흔들림이라 두 모델 비교 잣대가 아닙니다. 채택 기준은 context의 decision_rule에 미리 정해져 있습니다. 여러 안을 시도할수록 최고점은 운으로 높아지므로 기준에 아슬아슬하게 걸친 안은 더 의심하세요. 이미 채점된 안과 같은 spec은 다시 채점하지 않습니다.
- 개선안마다 근거(rationale)와 실제로 받은 근거 ID(evidence_ids)를 적으세요. 한국어로 간결하게, 최종 답은 지정된 JSON 형식으로만 내세요.'''

IMPROVER_FOCUS = {'A': '당신은 에이전트 A입니다. 주로 변수 쪽(오답 구간, 계절성, 다른 서점 신호, 판매 추세)에서 개선 가설을 세웁니다.',
                  'B': '당신은 에이전트 B입니다. 주로 모델 설정·학습 기간·불필요한 변수 제거·데이터 정제 쪽에서 개선 가설을 세웁니다.'}
FINAL_ROLE = PROCESS_COMMON + '''

당신은 적대적 검토자입니다. 지금까지의 개선안과 검증 점수(proposals, scores, debate)를 읽고 최종안을 고르세요.
- 반박이 임무입니다. 채택 기준(context의 decision_rule)을 넘지 못한 안은 고르지 마세요. 기준을 넘은 안이 여럿이면 더 단순한 안(변수·조치가 적은 안)을 우선하고, AP가 크게 떨어지는 등 다른 지표가 반대 방향이면 그 점도 따지세요. 여러 안을 시도할수록 최고점은 운으로 높아진다는 점도 고려하세요.
- 기준을 넘는 안이 없으면 choice에 baseline을 적으세요. 있으면 그 proposal_id를 적으세요.
- 탈락시킨 주요 안과 이유를 rejected에, 판단에 쓴 수치 주장을 claims에 적으세요. claims의 value는 근거 ID의 조회 결과에 실제로 있는 숫자여야 하고, 코드가 대조합니다.'''


def register_roles():
    ag.TOOLS['describe_db'] = ('공정 DB의 테이블·열 목록과 설명', ag._strict({}))
    ag.TOOLS['query_db'] = ('공정 DB에 읽기 전용 SELECT 하나를 실행한다(최대 100행)', ag._strict({'sql': {'type': 'string'}}))
    for who in 'AB':
        for r in range(1, ROUNDS + 1):
            role = f'improver_{who}_r{r}'
            ag.ROLES[role] = PROCESS_COMMON + '\n\n' + IMPROVER_FOCUS[who]
            ag.OUT[role] = IMPROVER_OUT
            ag.ROLE_TOOLS[role] = ['describe_db', 'query_db']
            ag.PREFIX[role] = f'{who}{r}'
    ag.ROLES['final_reviewer'] = FINAL_ROLE
    ag.OUT['final_reviewer'] = FINAL_OUT
    ag.ROLE_TOOLS['final_reviewer'] = ['describe_db', 'query_db']
    ag.PREFIX['final_reviewer'] = 'R'


register_roles()


def db_fingerprint(db):
    h = hashlib.sha256()
    for t in ('context', 'baseline_val', 'proposals', 'scores', 'debate'):
        h.update(json.dumps(db.execute(f'SELECT * FROM {t} ORDER BY 1').fetchall(), ensure_ascii=False, default=str).encode())
    return h.hexdigest()[:16]


def numbers(obj):
    """조회 결과 안의 숫자 값만 모은다(열 이름 같은 문자열 속 숫자는 세지 않는다)."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        t = obj.strip()
        if re.fullmatch(r'-?\d+(\.\d+)?', t):
            return [float(t)]
        if t[:1] in '[{':  # 셀에 JSON으로 저장된 목록(예: fold_deltas)
            try:
                return numbers(json.loads(t))
            except ValueError:
                return []
        return []
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in numbers(v)]
    if isinstance(obj, (list, tuple)):
        return [n for v in obj for n in numbers(v)]
    return []


def verify_claims(claims, evidence):
    out = []
    for c in claims:
        ev = evidence.get(c['evidence_id'])
        nums = numbers(ev['result']) if ev else []
        ok = any(abs(c['value'] - n) <= 1e-3 + 5e-3 * abs(n) or abs(c['value'] - n * 100) <= 0.05 + 5e-3 * abs(n * 100) for n in nums)
        out.append(dict(c, verified=int(ok), note='' if ok else ('근거 ID 없음' if ev is None else '조회 결과에 없는 값')))
    return out


def run_store(lab, agents, db_path, say=print):
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    say(f'{lab.store}: 0단계 분석을 DB에 저장')
    base_scores = write_stage0(db, lab)
    ws = ProcessWS(db_path)
    specs = {'baseline': dict(BASE_SPEC)}

    def call(role, user):
        out, evidence, meta = agents.run(role, ws, user, db_fingerprint(db))
        for eid, e in evidence.items():
            db.execute('INSERT OR REPLACE INTO evidence VALUES(?,?,?,?,?)', (eid, role, e['tool'], json.dumps(e['input'], ensure_ascii=False),
                                                                             json.dumps(e['result'], ensure_ascii=False, default=str)))
        db.execute('INSERT INTO agent_log VALUES(?,?,?,?,?)', (role, meta['seconds'], meta['cost_usd'], json.dumps(meta.get('tool_calls', [])), int(meta.get('cached', False))))
        db.commit()
        say(f'{lab.store}: {role} 완료 · {meta["seconds"]}s · ${meta["cost_usd"]:.2f}{" (저장분)" if meta.get("cached") else ""}')
        return out, evidence

    all_evidence = {}
    for r in range(1, ROUNDS + 1):
        limit = 3 if r == 1 else 2
        outs = {}
        for who in 'AB':
            if r == 1:
                user = f'라운드 1(경쟁): DB를 읽고 검증 p20을 올릴 개선안을 최대 {limit}개 내세요. 상대 에이전트의 안은 아직 없습니다. critique는 빈 문자열로 두세요.'
            else:
                user = (f'라운드 {r}(토론, 수정 {r - 1}/2회): proposals·scores·debate 테이블에서 지금까지의 모든 안과 검증 점수를 확인하세요. '
                        f'critique에 상대 에이전트 안의 강점·약점을 근거 ID와 함께 적고, 상대 안을 반박하거나 결합한 수정안을 최대 {limit}개 내세요. '
                        '이미 채점된 안과 같은 안은 내지 마세요.')
            outs[who], ev = call(f'improver_{who}_r{r}', user)
            all_evidence.update(ev)
        for who, out in outs.items():  # 두 에이전트가 모두 답한 뒤에 채점·공개한다(같은 라운드 안에서는 서로 모름)
            db.execute('INSERT INTO debate VALUES(?,?,?,?)', (r, who, out.get('critique', ''), out.get('notes', '')))
            for i, p in enumerate(out['proposals'][:limit], 1):
                pid = f'R{r}-{who}{i}'
                specs[pid] = score_proposal(db, lab, base_scores, pid, r, who, p)
        best = db.execute('SELECT proposal_id, val_p20, delta_p20 FROM scores WHERE round=? ORDER BY val_p20 DESC LIMIT 1', (r,)).fetchone()
        say(f'{lab.store}: 라운드 {r} 채점 완료 · 최고 {best}')
    out, ev = call('final_reviewer', '최종 검토: 모든 개선안과 점수를 검토해 최종안을 고르세요.')
    all_evidence.update(ev)
    choice = out['choice'] if out['choice'] in specs else 'baseline'
    passing = [r[0] for r in db.execute('SELECT proposal_id FROM scores WHERE prob_better >= ? AND delta_p20 > 0 AND folds_improved >= ? ORDER BY proposal_id',
                                        (PROB_RULE, MIN_FOLDS))]
    db.execute('CREATE TABLE IF NOT EXISTS rule_check(choice TEXT, passing TEXT, choice_passes INTEGER)')
    db.execute('INSERT INTO rule_check VALUES(?,?,?)', (choice, json.dumps(passing), int(choice == 'baseline' and not passing or choice in passing)))
    say(f'{lab.store}: 검토 선택 {choice} · 기준 통과 안 {passing or "없음"}')
    db.execute('INSERT INTO review VALUES(?,?,?)', (choice, out['reasons'], json.dumps(out, ensure_ascii=False)))
    for c in verify_claims(out['claims'], all_evidence):
        db.execute('INSERT INTO claims VALUES(?,?,?,?,?,?)', ('final_reviewer', c['text'], c['evidence_id'], c['value'], c['verified'], c['note']))
    # 단계별 대표안: 검증 점수로만 고른다(시험 구간은 이 선택에 쓰지 않는다)
    best = lambda where: (db.execute(f'SELECT proposal_id FROM scores WHERE val_p20 IS NOT NULL AND {where} ORDER BY val_p20 DESC, proposal_id LIMIT 1').fetchone() or ['baseline'])[0]
    stages = {'기준 모델': 'baseline', '에이전트 1명 (A 라운드1 최고)': best("round=1 AND agent='A'"), '경쟁 (라운드1 최고)': best('round=1'),
              '토론 (전 라운드 검증 최고, 검토 없음)': best('1=1'), '적대적 검토 (최종안)': choice}
    base_test = None
    for stage, pid in stages.items():
        s = specs[pid]
        ev_val, sc_val, _ = lab.evaluate(s, 'val')
        ev_t, sc_t, _ = lab.evaluate(s, 'test')
        vp, (tp, ta) = p20_ap(ev_val, sc_val)[0], p20_ap(ev_t, sc_t)
        base_test = tp if base_test is None else base_test
        db.execute('INSERT INTO final_results VALUES(?,?,?,?,?,?)', (stage, pid, vp, tp, ta, tp - base_test))
    for r in lab.recommend(specs[choice]):
        db.execute('INSERT INTO recommendations VALUES(?,?,?,?,?)', (r['rank'], r['isbn'], r['title'], r['score'], r['cutoff']))
    db.commit()
    say(f'{lab.store}: 최종안 {choice} · ' + ', '.join(f'{k} {v}' for k, v in stages.items()))
    return db


# ---------------------------------------------------------------- 보고서
pc = lambda v: '–' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v * 100:.1f}%'
pp = lambda v: '–' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v * 100:+.1f}%p'


def render(dbs, config):
    t = common.table
    parts = ['<h1>에이전트 경쟁·토론으로 재주문 예측 개선</h1>',
             f'<p class="note">실행 {html.escape(config["run_id"])} · 모델 {html.escape(config["model"])} · 백엔드 {config["backend"]} · '
             f'비용 ${config["cost_usd"]:.2f}{" (구독 사용: API 환산 추정치)" if config["backend"] == "subscription" else ""}. '
             '선택은 모두 검증 구간에서 했고, 시험 구간은 최종 평가에서 한 번만 썼다.</p>']
    for store, path in dbs.items():
        db = sqlite3.connect(path)
        q = lambda sql: [dict(zip([d[0] for d in c.description], r)) for c in [db.execute(sql)] for r in c.fetchall()]
        bv = {r['metric']: r['value'] for r in q('SELECT * FROM baseline_val')}
        parts.append(f'<section><h2>{html.escape(store)}</h2><p>기준 모델 검증 p20 {pc(bv.get("p20"))} (부트스트랩 표준편차 {pc(bv.get("p20_sd"))}), 판매 규칙 {pc(bv.get("rule_p20"))}.</p>'
                     '<h3>단계별 결과</h3>' + t(q('SELECT * FROM final_results'), [('stage', '단계', None), ('proposal_id', '안', None), ('val_p20', '검증 p20(3구간)', pc),
                                                                              ('test_p20', '시험 p20', pc), ('test_delta_vs_base', '시험 기준 대비', pp), ('test_ap', '시험 AP', pc)]))
        rows = q('SELECT p.proposal_id, p.name, p.rationale, p.problems, s.val_p20, s.delta_p20, s.delta_sd, s.prob_better, s.n_features, s.fold_deltas, s.folds_improved FROM proposals p JOIN scores s USING(proposal_id) ORDER BY p.proposal_id')
        parts.append('<h3>개선안 점수표 (검증 구간)</h3>' + t([dict(r, rationale=(r['rationale'] or '')[:140]) for r in rows],
                     [('proposal_id', '안', None), ('name', '이름', None), ('val_p20', 'p20', pc), ('delta_p20', '기준 대비', pp), ('delta_sd', '표준편차', pc),
                      ('prob_better', '개선 확률', pc), ('folds_improved', '나은 구간', str), ('fold_deltas', '구간별 차이', None), ('n_features', '변수 수', str), ('rationale', '근거', None), ('problems', '문제', None)]))
        parts.append('<h3>토론 기록</h3>' + t([dict(r, critique=(r['critique'] or '')[:400]) for r in q('SELECT * FROM debate WHERE round > 1')],
                                              [('round', '라운드', str), ('agent', '에이전트', None), ('critique', '상대 안 평가', None)]))
        rv = q('SELECT * FROM review')[0]
        rej = json.loads(rv['payload']).get('rejected', [])
        parts.append(f'<h3>적대적 검토</h3><p><b>최종안: {html.escape(rv["choice"])}</b> — {html.escape(rv["reasons"])}</p>' +
                     t(rej, [('proposal_id', '탈락 안', None), ('reason', '이유', None)]) +
                     t(q('SELECT text, evidence_id, value, verified, note FROM claims'), [('text', '주장', None), ('evidence_id', '근거', None), ('value', '값', str),
                                                                                         ('verified', 'DB 대조', lambda v: '일치' if v else '불일치'), ('note', '비고', None)]))
        log = q('SELECT role, seconds, cost_usd, cached FROM agent_log')
        parts.append('<h3>에이전트 실행</h3>' + t(log, [('role', '역할', None), ('seconds', '초', str), ('cost_usd', '비용', lambda v: f'${v:.2f}'), ('cached', '저장분', str)]))
        rec = q('SELECT * FROM recommendations ORDER BY rank LIMIT 30')
        if rec:
            parts.append(f'<h3>다음 달 재주문 예상 도서 (최종안, 기준일 {html.escape(rec[0]["cutoff"])}, 상위 30)</h3>' +
                         t(rec, [('rank', '순위', str), ('title', '도서', None), ('score', '점수', lambda v: f'{v:.3f}')]))
        parts.append('</section>')
        db.close()
    return f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>에이전트 개선 공정</title><style>{common.CSS}</style><body>{"".join(parts)}</body></html>'


def load_events(store):
    raw = ei.STORES[store]['load']()[:4]
    _, drop = kyobo.quality(*raw)
    return kyobo.clean(*raw, drop, True)[1]


def main():
    import shutil
    p = argparse.ArgumentParser()
    p.add_argument('--store', choices=[*ei.STORES, 'all'], default='all')
    p.add_argument('--model', default='claude-opus-5-5')
    p.add_argument('--effort', default='medium', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    p.add_argument('--backend', choices=['subscription', 'api'], default='subscription')
    p.add_argument('--no-llm', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT.parent / '결과' / '개선')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = sqlite3.connect(args.output.parent / 'agent.sqlite')
    if args.no_llm:
        agents = None
    elif args.backend == 'api':
        agents = ag.ApiAgents(ag.make_client(), args.model, args.effort, cache)
    else:
        agents = ag.SubscriptionAgents(args.model, args.effort, cache)
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    out = args.output / run_id
    out.mkdir()
    logf = open(out / 'run.log', 'a', encoding='utf8')

    def say(msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n'); logf.flush()

    config = dict(run_id=run_id, model=args.model, effort=args.effort, backend='none' if args.no_llm else args.backend, rounds=ROUNDS, cost_usd=0.0)
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding='utf8')
    started, dbs = time.time(), {}
    for store in (ei.STORES if args.store == 'all' else [args.store]):
        lab = Lab(store, load_events(OTHER[store]))
        path = out / f'process_{store}.sqlite'
        if agents is None:
            db = sqlite3.connect(path); db.executescript(SCHEMA); write_stage0(db, lab); db.close()
            say(f'{store}: 0단계만 저장 (--no-llm)')
            continue
        run_store(lab, agents, path, say).close()
        dbs[store] = path
        config['cost_usd'] = agents.cost
        (out / 'index.html').write_text(render(dbs, config), encoding='utf8')
    config.update(seconds=round(time.time() - started), cost_usd=agents.cost if agents else 0.0)
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding='utf8')
    if dbs:
        (out / 'index.html').write_text(render(dbs, config), encoding='utf8')
    say(f'끝: {config["seconds"]}s · 비용 ${config["cost_usd"]:.2f}')
    logf.close()
    files = sorted(f for f in out.rglob('*') if f.is_file() and f.name != 'manifest.json')
    (out / 'manifest.json').write_text(json.dumps(dict(run_id=run_id, raw_data_included=False,
        note='process_*.sqlite의 recommendations 테이블에는 최종 추천 도서명·ISBN이 들어 있다(에이전트에게는 공개하지 않음)',
        files=[dict(path=f.relative_to(out).as_posix(), bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]),
        ensure_ascii=False, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
