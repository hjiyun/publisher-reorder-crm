"""LLM 멀티 에이전트 데이터 검토: 독립 분석 A·B → 토론 조정 → 적대적 검토.

에이전트는 원본 행을 보지 않는다. 읽기 전용 집계 도구(건수·분포·월별 합계)만 호출하고, 결과마다 근거 ID를 받는다.
에이전트가 고를 수 있는 것은 미리 정한 정제 조치 목록(ACTIONS)과 그 매개변수뿐이고, 실제 적용은 코드가 한다.
근거 ID가 도구 결과에 없는 조치는 버린다(common.review와 같은 원칙).

역할
- 분석가 A: 표별 분포·키 중복·수량 이상값
- 분석가 B: 날짜·문서번호·표 간 흐름의 일관성
- 조정자: A·B 제안을 합치고 엇갈리는 부분을 도구로 확인한다(조치 미리보기 사용 가능)
- 적대적 검토자: 조치마다 정상 행을 얼마나 건드리는지 미리보기로 확인하고 승인·거부·축소한다
"""
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / '교보' / '코드'))

from common import SCHEMA, isbn13_valid  # noqa: E402

TABLES = ['rcvd', 'rtgd', 'sales']
TABLE_KO = {'rcvd': '재주문 이벤트(교보 입하 / 예스24 발주)', 'rtgd': '반품·반출', 'sales': '월별 판매'}
KEYS = {'rcvd': ['isbn', 'date', 'qty', 'doc', 'center', 'buy'], 'rtgd': ['isbn', 'date', 'qty', 'doc', 'reason'], 'sales': ['isbn', 'month']}
ALL_KEYS = sorted({k for v in KEYS.values() for k in v} | {'ALL'})
SALE_CH = ['store', 'online', 'interpark', 'corp']
ACTIONS = {
    'drop_exact_duplicates': '모든 열이 같은 행을 첫 행만 남기고 제거',
    'drop_key_duplicates': 'keys 조합이 같은 행을 첫 행만 남기고 제거',
    'drop_nonpositive_qty': '수량 0 이하 행 제거',
    'drop_invalid_isbn': 'ISBN 체크 숫자 오류이거나 도서정보에 없는 행 제거',
    'rescale_qty_outliers': '같은 도서 중앙값 대비 threshold배 이상인 수량을 10으로 나눔 (require_multiple_of_10이면 10의 배수만)',
    'drop_qty_outliers': '같은 도서 중앙값 대비 threshold배 이상인 행 제거',
    'drop_before_publication': '출간월 이전 기록 제거',
    'repair_dates_from_doc': '같은 문서번호(2행 이상) 안에서 문서 중앙 날짜와 max_days 넘게 다른 행의 날짜를 문서 중앙 날짜로 고침',
    'drop_doc_date_outliers': '문서 안 날짜 불일치 또는 앞뒤 문서번호 날짜와 max_days 넘게 다른 행 제거',
    'drop_isolated_events': '같은 도서의 판매가 앞뒤 window_months개월 안에 전혀 없는 이벤트·반품 행 제거',
    'flag_only': '자료를 바꾸지 않고 문제만 기록 (고칠 수 없는 누락 등)',
}
PRICES = {'claude-opus-5-5': (4.0, 20.0, 0.20, 5.0), 'claude-sonnet-5-5': (2.0, 10.0, 0.20, 2.5)}  # 입력, 출력, 캐시 읽기, 캐시 쓰기 ($/1M)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (pd.Period, pd.Timestamp)):
        return str(x)
    return x


# ---------------------------------------------------------------- 자료 보조
def month_of(df, name):
    return df.month if name == 'sales' else df.date.dt.to_period('M')


def qty_of(df, name):
    return df.total if name == 'sales' else df.qty


def qty_ratio(df, name):
    """같은 도서(3행 이상) 중앙값 대비 수량 배수. 판단할 수 없으면 NaN."""
    q = qty_of(df, name).astype(float)
    g = q.groupby(df.isbn)
    med = g.transform('median').where(g.transform('size') >= 3)
    return q / med.where(med > 0)


def doc_reference(df):
    """행마다 '그 문서의 기준 날짜'. 문서 안 최빈 날짜를 쓰고, 동률이면 앞뒤 문서번호(21개) 날짜 중앙값에 가까운 쪽을 고른다.
    그래도 정할 수 없으면 NaT(고치지 않음)."""
    d = df.dropna(subset=['date'])
    ref = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    if d.empty:
        return ref, pd.Series(np.nan, index=df.index)
    key = pd.to_numeric(d.doc.astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    day = (d.date - pd.Timestamp('2000-01-01')).dt.days.astype(float)
    docs = day.groupby(key).median().sort_index()
    near = docs.rolling(21, center=True, min_periods=5).median() if len(docs) >= 21 else pd.Series(np.nan, index=docs.index)
    near_row = key.map(near)
    out = {}
    for doc, idx in d.groupby('doc').groups.items():
        counts = d.loc[idx, 'date'].value_counts()
        if len(counts) == 1 or counts.iloc[0] > counts.iloc[1]:
            out[doc] = counts.index[0]
            continue
        r = near_row.loc[idx].iloc[0]
        tied = counts.index[counts == counts.iloc[0]]
        out[doc] = pd.NaT if pd.isna(r) else min(tied, key=lambda c: abs((c - pd.Timestamp('2000-01-01')).days - r))
    ref.loc[d.index] = d.doc.map(out).to_numpy()
    return ref, pd.Series(near_row, index=d.index).reindex(df.index)


def doc_flags(df, max_days):
    """(문서 안 날짜 불일치, 앞뒤 문서 날짜와 불일치) 행 플래그."""
    ref, near = doc_reference(df)
    size = df.groupby('doc').doc.transform('size')
    within = (size >= 2) & ref.notna() & ((df.date - ref).abs().dt.days > max_days)
    day = (df.date - pd.Timestamp('2000-01-01')).dt.days.astype(float)
    neighbor = ((day - near).abs() > max_days).fillna(False) & (size == 1)
    return within.fillna(False).astype(bool), neighbor.astype(bool)


def doc_order_corr(df):
    d = df.dropna(subset=['date'])
    key = pd.to_numeric(d.doc.astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    ok = key.notna()
    return float(pd.Series(key[ok].to_numpy()).corr(pd.Series(d.date[ok].rank().to_numpy()), method='spearman')) if ok.sum() > 10 else None


def isolated(df, name, sales, window):
    """이벤트·반품 행 가운데 같은 도서 판매가 앞뒤 window개월 안에 없는 행."""
    if name == 'sales':
        return pd.Series(False, index=df.index)
    active = set(zip(sales[qty_of(sales, 'sales') > 0].isbn, sales[qty_of(sales, 'sales') > 0].month))
    m = month_of(df, name)
    hit = pd.Series(False, index=df.index)
    for k in range(-window, window + 1):
        hit |= pd.Series([(i, p + k) in active for i, p in zip(df.isbn, m)], index=df.index, dtype=bool)
    return ~hit & m.notna()


# ---------------------------------------------------------------- 집계 도구 (원본 행·ISBN은 돌려주지 않는다)
class Workspace:
    def __init__(self, books, tables):
        self.books = books
        self.t = {k: v for k, v in tables.items()}
        self.pub = books.set_index('isbn').pub

    def table_overview(self, table):
        df, q = self.t[table], qty_of(self.t[table], table)
        m = month_of(df, table)
        big = q[q >= 10]
        return dict(table=TABLE_KO[table], rows=len(df), books=int(df.isbn.nunique()), first_month=m.min(), last_month=m.max(),
                    qty_quantiles={p: float(q.quantile(p / 100)) for p in (1, 25, 50, 75, 90, 99)}, qty_max=float(q.max()),
                    nonpositive_qty=int((q <= 0).sum()), multiple_of_10_share_among_qty_ge_10=float((big % 10 == 0).mean()) if len(big) else None,
                    monthly_rows=m.value_counts().sort_index().to_dict(), monthly_qty=q.groupby(m).sum().sort_index().to_dict())

    def key_uniqueness(self, table, keys):
        df = self.t[table]
        if keys == ['ALL'] or 'ALL' in keys:
            cols = list(df.columns)
        else:
            bad = [k for k in keys if k not in KEYS[table]]
            if bad or not keys:
                raise ValueError(f'{table}에서 쓸 수 있는 키: {KEYS[table]} 또는 ALL')
            cols = keys
        size = df.groupby(cols, dropna=False).size()
        dup = size[size > 1]
        return dict(table=TABLE_KO[table], keys=cols, duplicate_groups=int(len(dup)), extra_rows=int((dup - 1).sum()),
                    share_of_rows=float((dup - 1).sum() / len(df)) if len(df) else 0.0, group_size_histogram=dup.value_counts().sort_index().to_dict())

    def qty_ratio_outliers(self, table, threshold):
        df = self.t[table]
        r = qty_ratio(df, table)
        q = qty_of(df, table)
        hit = r >= threshold
        bins = pd.cut(r[r >= 2], [2, 3, 5, 8, 10, 15, np.inf], right=False).value_counts().sort_index()
        return dict(table=TABLE_KO[table], threshold=threshold, rows_at_or_above=int(hit.sum()), share_of_rows=float(hit.mean()),
                    of_which_multiple_of_10=int((hit & (q % 10 == 0)).sum()), rows_not_assessable=int(r.isna().sum()),
                    ratio_histogram={str(k): int(v) for k, v in bins.items()})

    def temporal_consistency(self, table):
        df = self.t[table]
        m = month_of(df, table)
        pub_m = df.isbn.map(self.pub).dt.to_period('M')
        out = dict(table=TABLE_KO[table], before_publication_month=int((m < pub_m).sum()), unknown_publication=int(pub_m.isna().sum()),
                   missing_date=int(m.isna().sum()))
        if table != 'sales':
            out['isolated_from_sales'] = {f'±{w}개월': int(isolated(df, table, self.t['sales'], w).sum()) for w in (1, 2, 3)}
        else:
            ev = set(zip(self.t['rcvd'].isbn, month_of(self.t['rcvd'], 'rcvd')))
            out['sales_rows_without_event_within_±3개월'] = int(sum(all((i, p + k) not in ev for k in range(-3, 4)) for i, p in zip(df.isbn, m)))
        return out

    def doc_date_consistency(self, table):
        if table == 'sales':
            raise ValueError('판매표에는 문서번호가 없다')
        df = self.t[table]
        out = dict(table=TABLE_KO[table], docs=int(df.doc.nunique()), rows_in_multi_row_docs=int((df.groupby('doc').doc.transform('size') >= 2).sum()),
                   doc_number_vs_date_spearman=doc_order_corr(df))
        for days in (90, 180, 300):
            w, n = doc_flags(df, days)
            out[f'over_{days}d'] = dict(within_doc=int(w.sum()), vs_neighbor_docs=int(n.sum()), either=int((w | n).sum()))
        return out

    def referential_integrity(self, table):
        df = self.t[table]
        return dict(table=TABLE_KO[table], invalid_isbn_check_digit=int((~isbn13_valid(df.isbn)).sum()), isbn_not_in_book_list=int((~df.isbn.isin(self.books.isbn)).sum()))

    def cross_table_flow(self):
        tot = {n: qty_of(self.t[n], n).groupby(month_of(self.t[n], n)).sum() for n in TABLES}
        f = pd.DataFrame(tot).fillna(0)
        z = (f - f.rolling(7, center=True, min_periods=3).median()) / f.std().replace(0, np.nan)
        return dict(monthly_corr={'이벤트-판매': float(f.rcvd.corr(f.sales)), '반품-이벤트': float(f.rtgd.corr(f.rcvd))},
                    months_with_large_deviation={TABLE_KO[n]: {str(k): float(v) for k, v in z[n][z[n].abs() > 2.5].items()} for n in TABLES},
                    total_qty={TABLE_KO[n]: float(f[n].sum()) for n in TABLES})

    def preview_action(self, action):
        before = {n: d for n, d in self.t.items()}
        after, log = apply_actions(self.books, before, [action])
        t = action['table']
        b, a = before[t], after[t]
        removed = b.index.difference(a.index)
        common = b.index.intersection(a.index)
        col = 'total' if t == 'sales' else 'qty'
        changed = (b.loc[common, col] != a.loc[common, col]) | ((b.loc[common, 'date'] != a.loc[common, 'date']) if t != 'sales' else False)
        mb, ma = qty_of(b, t).groupby(month_of(b, t)).sum(), qty_of(a, t).groupby(month_of(a, t)).sum()
        diff = (ma.reindex(mb.index.union(ma.index), fill_value=0) - mb.reindex(mb.index.union(ma.index), fill_value=0))
        return dict(action=action['action'], table=TABLE_KO[t], rows_removed=int(len(removed)), rows_modified=int(changed.sum()),
                    share_of_table=float((len(removed) + changed.sum()) / len(b)) if len(b) else 0.0,
                    books_touched=int(pd.concat([b.loc[removed, 'isbn'], b.loc[common[changed.to_numpy()], 'isbn']]).nunique()),
                    total_qty_change=float(qty_of(a, t).sum() - qty_of(b, t).sum()), months_changed=int((diff != 0).sum()),
                    largest_monthly_change=float(diff.abs().max()) if len(diff) else 0.0)


# ---------------------------------------------------------------- 정제 조치 (코드가 적용한다. 인덱스는 유지)
def apply_actions(books, tables, actions):
    t = {k: v.copy() for k, v in tables.items()}
    log = []
    for a in actions:
        name, kind = a['table'], a['action']
        df = t[name]
        n0 = len(df)
        if kind == 'flag_only':
            log.append(dict(action=kind, table=name, removed=0, modified=0))
            continue
        if kind == 'drop_exact_duplicates':
            df = df[~df.duplicated(keep='first')]
        elif kind == 'drop_key_duplicates':
            keys = [k for k in a.get('keys', []) if k in KEYS[name]] or list(df.columns)
            df = df[~df.duplicated(subset=keys, keep='first')]
        elif kind == 'drop_nonpositive_qty':
            df = df[qty_of(df, name) > 0]
        elif kind == 'drop_invalid_isbn':
            df = df[isbn13_valid(df.isbn) & df.isbn.isin(books.isbn)]
        elif kind in ('rescale_qty_outliers', 'drop_qty_outliers'):
            hit = qty_ratio(df, name) >= max(float(a.get('threshold') or 0), 2.0)
            if a.get('require_multiple_of_10'):
                hit &= qty_of(df, name) % 10 == 0
            if kind == 'drop_qty_outliers':
                df = df[~hit]
            else:
                df = df.copy()
                cols = SALE_CH + ['total'] if name == 'sales' else ['qty']
                df.loc[hit, cols] = df.loc[hit, cols] / 10
        elif kind == 'drop_before_publication':
            pub_m = df.isbn.map(books.set_index('isbn').pub).dt.to_period('M')
            df = df[~(month_of(df, name) < pub_m)]
        elif kind in ('repair_dates_from_doc', 'drop_doc_date_outliers'):
            if name == 'sales':
                raise ValueError('판매표에는 문서번호가 없다')
            days = max(int(a.get('max_days') or 0), 30)
            within, neighbor = doc_flags(df, days)
            if kind == 'drop_doc_date_outliers':
                df = df[~(within | neighbor)]
            else:
                df = df.copy()
                df.loc[within, 'date'] = doc_reference(df)[0][within]
        elif kind == 'drop_isolated_events':
            df = df[~isolated(df, name, t['sales'], max(int(a.get('window_months') or 0), 1))]
        else:
            raise ValueError(f'알 수 없는 조치: {kind}')
        common = df.index.intersection(t[name].index)
        col = 'total' if name == 'sales' else 'qty'
        modified = int(((t[name].loc[common, col] != df.loc[common, col]) |
                        ((t[name].loc[common, 'date'] != df.loc[common, 'date']) if name != 'sales' else False)).sum())
        log.append(dict(action=kind, table=name, removed=n0 - len(df), modified=modified))
        t[name] = df
    return t, log


# ---------------------------------------------------------------- 도구·출력 스키마
def _strict(props, required=None):
    return {'type': 'object', 'properties': props, 'required': list(props) if required is None else required, 'additionalProperties': False}


TABLE_ENUM = {'type': 'string', 'enum': TABLES}
ACTION_SCHEMA = _strict({
    'action': {'type': 'string', 'enum': list(ACTIONS)},
    'table': TABLE_ENUM,
    'keys': {'type': 'array', 'items': {'type': 'string', 'enum': ALL_KEYS}},
    'threshold': {'type': 'number'},
    'window_months': {'type': 'integer'},
    'max_days': {'type': 'integer'},
    'require_multiple_of_10': {'type': 'boolean'},
    'rationale': {'type': 'string'},
    'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
})
TOOLS = {
    'table_overview': ('표의 행 수, 기간, 수량 분위수, 0 이하 수량, 월별 행 수·수량 합계', _strict({'table': TABLE_ENUM})),
    'key_uniqueness': ('키 조합 중복 건수. keys=["ALL"]이면 모든 열이 같은 중복', _strict({'table': TABLE_ENUM, 'keys': {'type': 'array', 'items': {'type': 'string', 'enum': ALL_KEYS}}})),
    'qty_ratio_outliers': ('같은 도서 중앙값 대비 threshold배 이상인 수량 건수와 배수 분포', _strict({'table': TABLE_ENUM, 'threshold': {'type': 'number'}})),
    'temporal_consistency': ('출간 전 기록, 날짜 누락, 판매와 동떨어진 이벤트 건수', _strict({'table': TABLE_ENUM})),
    'doc_date_consistency': ('문서번호 안·앞뒤 문서와 날짜가 어긋나는 건수 (재주문 이벤트·반품만)', _strict({'table': TABLE_ENUM})),
    'referential_integrity': ('ISBN 체크 숫자 오류, 도서정보에 없는 ISBN 건수', _strict({'table': TABLE_ENUM})),
    'cross_table_flow': ('표 간 월별 합계 상관과 추세에서 크게 벗어난 달', _strict({})),
    'preview_action': ('정제 조치를 적용하면 몇 행이 지워지거나 바뀌는지 미리 계산 (자료는 바뀌지 않음)', _strict({'action': ACTION_SCHEMA})),
}
FINDING = _strict({'issue': {'type': 'string'}, 'table': {'type': 'string', 'enum': TABLES + ['all']}, 'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
                   'affected_rows_estimate': {'type': 'integer'}, 'severity': {'type': 'string', 'enum': ['high', 'medium', 'low']}, 'fixable': {'type': 'boolean'}})
ANALYST_OUT = _strict({'findings': {'type': 'array', 'items': FINDING}, 'actions': {'type': 'array', 'items': ACTION_SCHEMA}})
MOD_ACTION = _strict({**ACTION_SCHEMA['properties'], 'support': {'type': 'string', 'enum': ['both', 'A', 'B', 'moderator']}})
MOD_OUT = _strict({'actions': {'type': 'array', 'items': MOD_ACTION},
                   'dropped': {'type': 'array', 'items': _strict({'proposal': {'type': 'string'}, 'reason': {'type': 'string'}})},
                   'unresolved': {'type': 'array', 'items': {'type': 'string'}}})
REVIEW_OUT = _strict({'verdicts': {'type': 'array', 'items': _strict({
    'index': {'type': 'integer'}, 'verdict': {'type': 'string', 'enum': ['approve', 'reject', 'modify']}, 'reason': {'type': 'string'},
    'evidence_ids': {'type': 'array', 'items': {'type': 'string'}}, 'modified': ACTION_SCHEMA})}})


def tool_defs(names):
    return [{'name': n, 'description': TOOLS[n][0], 'strict': True, 'input_schema': TOOLS[n][1]} for n in names]


PROFILE_TOOLS = [n for n in TOOLS if n != 'preview_action']

COMMON = f'''당신은 출판사 데이터 품질 검토 에이전트입니다. 자료는 한 출판사가 서점 협력사 시스템(SCM)에서 받은 기록입니다.
- 표: rcvd = {TABLE_KO['rcvd']} (isbn, date, qty, doc=문서번호, center, buy), rtgd = {TABLE_KO['rtgd']} (isbn, date, qty, doc, reason), sales = {TABLE_KO['sales']} (isbn, month, 채널별 수량, total).
- 이 자료로 매월 1일 기준 "다음 30일 안에 서점이 재주문(rcvd 1건 이상)할 도서 상위 20%"를 예측합니다. 변수는 기준일 이전의 이벤트·반품·판매 합계와 횟수, 최근성입니다.
- 자료에는 입력·전송 과정의 오류가 섞여 있을 수 있습니다. 정상 기록을 지우면 실제 재주문 라벨이 사라져 예측이 나빠지므로, 고칠 수 있으면 지우기보다 고치고, 근거가 약하면 손대지 않는 편이 낫습니다.
- 당신은 원본 행을 볼 수 없습니다. 집계 도구만 쓸 수 있고, 도구 결과마다 근거 ID(evidence_id)가 붙습니다. 모든 주장과 조치에는 실제로 받은 근거 ID를 적으세요. 없는 ID를 적은 조치는 자동으로 버려집니다.
- 쓸 수 있는 정제 조치(action)와 뜻:
''' + '\n'.join(f'  · {k}: {v}' for k, v in ACTIONS.items()) + '''
- 조치에서 쓰지 않는 매개변수는 keys=[], threshold=0, window_months=0, max_days=0, require_multiple_of_10=false로 두세요.
- 한국어로 간결하게 쓰고, 최종 답은 지정된 JSON 형식으로만 내세요.'''

ROLES = {
    'analyst_A': COMMON + '\n\n역할: 분석가 A. 표마다 분포, 키 중복, 수량 이상값, 참조 무결성을 살펴 문제와 정제 조치를 제안하세요. 다른 분석가와 독립적으로 판단합니다.',
    'analyst_B': COMMON + '\n\n역할: 분석가 B. 날짜·문서번호의 일관성, 출간일과의 순서, 표 간 월별 흐름(이벤트·반품·판매)을 살펴 문제와 정제 조치를 제안하세요. 다른 분석가와 독립적으로 판단합니다.',
    'moderator': COMMON + '\n\n역할: 조정자. 두 분석가의 제안을 받아 하나의 조치 목록으로 합칩니다. 둘이 엇갈리면 도구와 preview_action으로 직접 확인해 정하세요. 중복 조치는 하나로 합치고, 근거가 부족한 조치는 dropped에 이유와 함께 적으세요.',
    'reviewer': COMMON + '\n\n역할: 적대적 검토자. 조정된 조치 목록을 반박하는 것이 임무입니다. 조치마다 preview_action으로 지워지거나 바뀌는 행 수를 확인하고, 그 행들이 정말 오류라고 볼 근거가 있는지 따지세요. 정상 기록을 많이 건드리거나 근거가 약하면 reject, 범위를 줄이면 되면 modify(줄인 조치를 modified에), 타당하면 approve 하세요. modify가 아니면 modified에는 원래 조치를 그대로 적으세요.',
}
OUT = {'analyst_A': ANALYST_OUT, 'analyst_B': ANALYST_OUT, 'moderator': MOD_OUT, 'reviewer': REVIEW_OUT}
ROLE_TOOLS = {'analyst_A': PROFILE_TOOLS, 'analyst_B': PROFILE_TOOLS, 'moderator': list(TOOLS), 'reviewer': list(TOOLS)}
PREFIX = {'analyst_A': 'A', 'analyst_B': 'B', 'moderator': 'M', 'reviewer': 'R'}
ROLE_BUILTINS = {}  # 역할별로 허용하는 기본 도구(예: 시험 일정 역할의 WebSearch). 그 밖의 역할은 기본 도구 없음


# ---------------------------------------------------------------- LLM 호출
def call_tool(ws, role, evidence, name, args):
    """도구 하나를 실행해 (돌려줄 문자열, 오류 여부)를 만든다. 성공하면 근거 ID를 붙여 evidence에 저장한다."""
    try:
        out = jsonable(getattr(ws, name)(**args))
    except Exception as e:  # 잘못된 키 등은 오류로 돌려주고 계속한다
        return f'오류: {e}', True
    eid = f'{PREFIX[role]}-E{len(evidence) + 1}'
    evidence[eid] = dict(tool=name, input=args, result=out)
    return json.dumps(dict(evidence_id=eid, **out), ensure_ascii=False), False


class Agents:
    """역할 실행과 결과 저장(같은 입력이면 다시 부르지 않음). 실제 호출은 하위 클래스의 _execute가 한다."""
    backend = 'base'

    def __init__(self, model='claude-opus-5-5', effort='medium', db=None, max_turns=12):
        self.model, self.effort, self.max_turns, self.db = model, effort, max_turns, db
        if db is not None:
            db.executescript(SCHEMA)
        self.cost = 0.0

    def run(self, role, ws, user, fingerprint, run_id=''):
        """역할 하나를 실행해 (출력 JSON, 근거 사전, 메타)를 돌려준다."""
        key = hashlib.sha256(json.dumps([self.model, self.effort, role, ROLES[role], user, ROLE_TOOLS[role], fingerprint], ensure_ascii=False).encode()).hexdigest()[:24]
        if self.db is not None:
            row = self.db.execute('SELECT payload FROM agent_outputs WHERE agent_run_id=?', (key,)).fetchone()
            if row:
                saved = json.loads(row[0])
                return saved['output'], saved['evidence'], dict(saved['meta'], cached=True)
        started, evidence = time.time(), {}
        output, extra = self._execute(role, ws, user, evidence)
        meta = dict(role=role, backend=self.backend, seconds=round(time.time() - started, 1), cached=False, **extra)
        self.cost += meta['cost_usd']
        if self.db is not None:
            self.db.execute('INSERT OR REPLACE INTO agent_outputs VALUES(?,?,?,?,?,?,?,?,?,?)',
                            (key, run_id, role, 0, meta['model'], user, json.dumps(dict(output=output, evidence=evidence, meta=meta), ensure_ascii=False),
                             time.strftime('%Y-%m-%d %H:%M:%S'), meta['seconds'], meta['cost_usd']))
            self.db.commit()
        return output, evidence, meta


def find_claude_cli():
    """Agent SDK가 실행할 claude 실행 파일. Windows에서는 npm의 claude.cmd를 쓸 수 없어 claude.exe를 찾는다.
    TBT_CLAUDE_CLI → PATH의 claude.exe → ~/.local/bin → Claude 데스크톱 앱에 들어 있는 최신 버전 순."""
    import shutil
    for c in (os.environ.get('TBT_CLAUDE_CLI'), shutil.which('claude.exe'), str(Path.home() / '.local' / 'bin' / 'claude.exe')):
        if c and Path(c).exists():
            return c
    found = list(Path(os.environ.get('APPDATA', '')).glob('Claude/claude-code/*/*/claude.exe'))
    version = lambda p: tuple(int(x) for x in p.parent.parent.name.split('.') if x.isdigit())
    if found:
        return str(max(found, key=lambda p: (version(p), p.stat().st_mtime)))
    if os.name != 'nt' and shutil.which('claude'):
        return shutil.which('claude')
    raise SystemExit('claude 실행 파일을 찾지 못했습니다. TBT_CLAUDE_CLI 환경 변수에 claude.exe 경로를 넣어 주세요.')


class SubscriptionAgents(Agents):
    """Claude Agent SDK로 호출한다. 이 컴퓨터의 Claude Code 로그인(구독)을 쓰므로 API 키가 필요 없다.
    비용은 SDK가 계산한 API 환산 추정치이고 실제 청구액이 아니다. 구독 사용량 한도에 포함된다."""
    backend = 'subscription'

    def __init__(self, *a, cli_path=None, **kw):
        super().__init__(*a, **kw)
        self.cli_path = cli_path or find_claude_cli()
        # 다른 Claude Code 세션 안에서 실행해도 그 세션의 연결 정보를 물려받지 않고 독립 로그인을 쓰게 한다.
        for k in [k for k in os.environ if k.startswith('CLAUDE_CODE_') or k in ('CLAUDECODE', 'ANTHROPIC_BASE_URL')]:
            os.environ.pop(k)

    def _execute(self, role, ws, user, evidence):
        import asyncio
        return asyncio.run(self._aexecute(role, ws, user, evidence))

    async def _aexecute(self, role, ws, user, evidence):
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolUseBlock, create_sdk_mcp_server, query, tool

        def make(name):
            async def handler(args):
                text, err = call_tool(ws, role, evidence, name, args)
                return {'content': [{'type': 'text', 'text': text}], **({'is_error': True} if err else {})}
            return tool(name, TOOLS[name][0], TOOLS[name][1])(handler)

        names, builtins = ROLE_TOOLS[role], ROLE_BUILTINS.get(role, [])
        options = ClaudeAgentOptions(
            system_prompt=ROLES[role], model=self.model, effort=self.effort,
            tools=list(builtins),  # 파일·셸 등 기본 도구는 끄고, 역할에 허용한 것(예: WebSearch)만 준다
            mcp_servers={'data': create_sdk_mcp_server(name='data', tools=[make(n) for n in names])} if names else {},
            allowed_tools=[f'mcp__data__{n}' for n in names] + list(builtins), permission_mode='dontAsk',
            setting_sources=[],  # 사용자 설정·메모리를 불러오지 않는다
            max_turns=self.max_turns * 2, cwd=str(ROOT), cli_path=self.cli_path,
            output_format={'type': 'json_schema', 'schema': OUT[role]})
        calls, result, model = [], None, self.model
        async for m in query(prompt=user, options=options):
            if isinstance(m, AssistantMessage):
                model = m.model or model
                calls += [b.name.split('__')[-1] for b in m.content if isinstance(b, ToolUseBlock)]
            elif isinstance(m, ResultMessage):
                result = m
        if result is None or result.is_error or result.structured_output is None:
            why = None if result is None else (result.errors or result.subtype)
            raise RuntimeError(f'{role}: 에이전트 실행 실패 ({why})')
        return result.structured_output, dict(model=model, cost_usd=round(result.total_cost_usd or 0.0, 4), tool_calls=calls, turns=result.num_turns)


def make_client():
    import anthropic
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise SystemExit('ANTHROPIC_API_KEY 환경 변수가 없습니다. 키를 직접 설정한 뒤 다시 실행하세요.')
    # Claude Code 같은 다른 도구의 ANTHROPIC_BASE_URL을 물려받지 않도록 기본 주소를 명시한다.
    return anthropic.Anthropic(base_url=os.environ.get('TBT_ANTHROPIC_BASE_URL', 'https://api.anthropic.com'))


class ApiAgents(Agents):
    """Anthropic API(키 필요)로 호출한다. 클라우드 재현 등 API 키를 쓸 때의 대안."""
    backend = 'api'

    def __init__(self, client, *a, fallback=True, **kw):
        super().__init__(*a, **kw)
        self.client, self.fallback = client, fallback

    def _price(self, usage):
        p = PRICES.get(self.model, PRICES['claude-opus-5-5'])
        get = lambda k: getattr(usage, k, 0) or 0
        return (get('input_tokens') * p[0] + get('output_tokens') * p[1] + get('cache_read_input_tokens') * p[2] + get('cache_creation_input_tokens') * p[3]) / 1e6

    def _create(self, **kw):
        if self.fallback:
            return self.client.beta.messages.create(betas=['server-side-fallback-2026-07-01'], fallbacks='default', **kw)
        return self.client.beta.messages.create(**kw)

    def _execute(self, role, ws, user, evidence):
        cost, calls, turn = 0.0, [], 0
        messages = [{'role': 'user', 'content': user}]
        tools = tool_defs(ROLE_TOOLS[role]) + ([{'type': 'web_search_20260209', 'name': 'web_search'}] if 'WebSearch' in ROLE_BUILTINS.get(role, []) else [])
        kw = dict(model=self.model, max_tokens=16000, **({'tools': tools} if tools else {}),
                  system=[{'type': 'text', 'text': ROLES[role], 'cache_control': {'type': 'ephemeral'}}],
                  output_config={'effort': self.effort, 'format': {'type': 'json_schema', 'schema': OUT[role]}})
        while True:
            final = turn >= self.max_turns
            resp = self._create(messages=messages, **kw, **({'tool_choice': {'type': 'none'}} if final else {}))
            cost += self._price(resp.usage)
            if resp.stop_reason == 'refusal':
                raise RuntimeError(f'{role}: 모델이 요청을 거절함')
            if resp.stop_reason == 'max_tokens':
                raise RuntimeError(f'{role}: 출력 길이 한도 도달')
            messages.append({'role': 'assistant', 'content': resp.content})
            if resp.stop_reason == 'pause_turn':
                continue
            uses = [b for b in resp.content if getattr(b, 'type', '') == 'tool_use']
            if resp.stop_reason != 'tool_use' or not uses:
                break
            results = []
            for b in uses:
                text, err = call_tool(ws, role, evidence, b.name, b.input)
                calls.append(b.name)
                results.append({'type': 'tool_result', 'tool_use_id': b.id, 'content': text, **({'is_error': True} if err else {})})
            messages.append({'role': 'user', 'content': results})
            turn += 1
        text = next((b.text for b in resp.content if getattr(b, 'type', '') == 'text'), '')
        return json.loads(text), dict(model=getattr(resp, 'model', self.model), cost_usd=round(cost, 4), tool_calls=calls, turns=turn)


def fingerprint(tables):
    h = hashlib.sha256()
    for n in TABLES:
        h.update(pd.util.hash_pandas_object(tables[n].astype(str), index=True).to_numpy().tobytes())
    return h.hexdigest()[:16]


def supported(action, evidence):
    ids = action.get('evidence_ids') or []
    return action['action'] == 'flag_only' or (bool(ids) and all(i in evidence for i in ids))


def review(agents, books, tables, context, run_id=''):
    """독립 분석 A·B → 조정 → 적대적 검토. 최종 조치와 단계별 기록을 돌려준다."""
    ws = Workspace(books, tables)
    fp = fingerprint(tables)
    brief = f'검토 대상: {context}. 자료를 살펴 문제를 찾고 정제 조치를 제안하세요.'
    a, ea, ma = agents.run('analyst_A', ws, brief, fp, run_id)
    b, eb, mb = agents.run('analyst_B', ws, brief, fp, run_id)
    ev = {**ea, **eb}
    proposals = json.dumps({'분석가 A': a, '분석가 B': b}, ensure_ascii=False)
    m, em, mm = agents.run('moderator', ws, f'검토 대상: {context}.\n두 분석가의 독립 제안입니다. 근거 ID는 그대로 인용할 수 있습니다.\n{proposals}', fp, run_id)
    ev.update(em)
    merged = [x for x in m['actions'] if supported(x, ev)]
    listing = json.dumps([dict(index=i, **x) for i, x in enumerate(merged)], ensure_ascii=False)
    r, er, mr = agents.run('reviewer', ws, f'검토 대상: {context}.\n조정된 조치 목록입니다. 각 index마다 판정을 내리세요.\n{listing}', fp, run_id)
    ev.update(er)
    final, verdicts = [], {v['index']: v for v in r['verdicts']}
    for i, x in enumerate(merged):
        v = verdicts.get(i)
        if v is None or v['verdict'] == 'reject':
            continue
        act = dict(v['modified']) if v['verdict'] == 'modify' else x
        if not act.get('evidence_ids'):  # 축소한 조치가 근거를 비워 두면 원래 조치의 근거를 잇는다
            act['evidence_ids'] = x.get('evidence_ids', [])
        if supported(act, ev):
            final.append({k: act[k] for k in ACTION_SCHEMA['properties']})
    unsupported = len(m['actions']) - len(merged)
    return dict(actions=final, analyst_A=a, analyst_B=b, moderator=m, reviewer=r, unsupported_dropped=unsupported, evidence=ev,
                meta=[ma, mb, mm, mr], cost_usd=round(sum(x['cost_usd'] for x in (ma, mb, mm, mr) if not x.get('cached')), 4))
