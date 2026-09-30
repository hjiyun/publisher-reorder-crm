"""교보문고 SCM 자료: 도서×월 단위 교보 재주문(입하) 예측과 도서 세분화.

실행:  python kyobo.py            입력 ../원본/, ../고객성향/ (구성은 ../README.md)
산출:  ../결과/<run_id>/index.html, ../결과/research.sqlite

설계 요약
- 기준일 T = 매월 1일. 변수는 T 이전 기록만 쓴다(판매는 T 이전 달까지의 월별 합계).
- 라벨: [T, T+30일) 안에 교보 입하(=교보가 주문해 받은 책)가 1건 이상인가.
- 대상: T 이전에 출간되었고, 직전 365일 안에 입하나 판매가 있었던 도서.
- 기준일: 판매 자료가 있는 마지막 달까지. 마지막 6개월 = 평가, 그 앞 6개월 = 검증, 나머지 = 학습.
- 조건 minimal / full(추가 품질 점검)을 같은 평가 도서·같은 정답으로 비교한다.
"""
import argparse
import hashlib
import html
import json
import math
import re
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import common as report
import crm_roi
from common import (CONTACT_FRACTION, GAIN_FRACTIONS, SCHEMA, AccountRFM, bootstrap_precision, isbn13_valid, make_models, metrics,
                    review, segment_table)

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / '원본'
HORIZON_DAYS, ACTIVE_DAYS, TEST_MONTHS, VAL_MONTHS = 30, 365, 6, 6
SALE_COLS = {'판매(영업점)': 'store', '판매(온라인)': 'online', '판매(인터파크)': 'interpark', '판매(법인)': 'corp'}
ymd = lambda s: pd.to_datetime(s.astype(str).str.strip(), format='%Y%m%d', errors='coerce')


# ---------------------------------------------------------------- 적재
def load(src):
    hashes = {}

    def rows(path):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.DataFrame(json.loads(path.read_text(encoding='utf8'))['rows'])

    b = rows(next((src / '도서정보').glob('*.json')))
    books = pd.DataFrame({'isbn': b.cmdt_code.astype(str).str.strip(), 'title': b.cmdt_name, 'pub': ymd(b.rlse_date),
                          'price': pd.to_numeric(b.cmdt_prce, errors='coerce'), 'genre': b.joName, 'status': b.cmdt_cdtn_code})
    r = rows(next((src / '입하상세').glob('*.json')))
    rcvd = pd.DataFrame({'isbn': r.cmdt_code.astype(str).str.strip(), 'date': ymd(r.rcvd_date), 'qty': pd.to_numeric(r.rcvd_qntt, errors='coerce'),
                         'buy': r.byng_dvsn_name, 'center': r.rdp_name, 'doc': r.undt_num.astype(str), 'rate': pd.to_numeric(r.byng_rate, errors='coerce')})
    t = rows(next((src / '반품조회').glob('*.json')))
    rtgd = pd.DataFrame({'isbn': t.cmdt_code.astype(str).str.strip(), 'date': ymd(t.rtgd_wrk_date), 'qty': pd.to_numeric(t.rtgd_qntt, errors='coerce'),
                         'buy': t.byng_dvsn_code_name, 'reason': t.rtgd_rsn_code, 'doc': t.rtgd_num.astype(str)})
    frames = []
    for path in sorted((src / '판매조회').glob('판매_*.xls')):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        x = pd.read_excel(path, header=3)
        x.columns = [str(c).replace('\n', '') for c in x.columns]
        x = x[x.ISBN.astype(str).str.fullmatch(r'\d{13}')].rename(columns=SALE_COLS)
        x = x.assign(isbn=x.ISBN.astype(str), month=pd.Period(path.stem.split('_')[1], 'M'))
        frames.append(x[['isbn', 'month', *SALE_COLS.values()]])
    sales = pd.concat(frames, ignore_index=True)
    for c in SALE_COLS.values():
        sales[c] = pd.to_numeric(sales[c], errors='coerce').fillna(0)
    sales['total'] = sales[list(SALE_COLS.values())].sum(axis=1)
    return books, rcvd, rtgd, sales, hashes


# ---------------------------------------------------------------- 품질 점검
def quality(books, rcvd, rtgd, sales):
    """(점검 결과 표, 테이블별 제외 플래그). ISO/IEC 25012 관점을 참고한 프로젝트 자체 규칙."""
    rows, drop = [], {}
    pub = books.set_index('isbn').pub

    def add(table, check, dim, rule, mask, n, action):
        rows.append(dict(table=table, check=check, dimension=dim, rule=rule, violations=int(mask.sum()), share=float(mask.sum() / n) if n else 0.0, action=action))

    for name, df in (('입하', rcvd), ('반품', rtgd)):
        f = pd.DataFrame({
            'date_invalid': df.date.isna(),
            'qty_nonpositive': ~(df.qty > 0),
            'isbn_invalid': ~isbn13_valid(df.isbn),
            'isbn_not_in_books': ~df.isbn.isin(books.isbn),
            'exact_duplicate': df.duplicated(keep='first'),
        })
        add(name, 'date_invalid', '완전성', '일자 누락·해석 불가', f.date_invalid, len(df), '제외 (두 조건 공통)')
        add(name, 'qty_nonpositive', '일관성', '수량 0 이하·누락', f.qty_nonpositive, len(df), 'full: 제외')
        add(name, 'isbn_invalid', '일관성', 'ISBN 형식·체크 숫자 오류', f.isbn_invalid, len(df), 'full: 제외')
        add(name, 'isbn_not_in_books', '일관성', '도서정보에 없는 ISBN', f.isbn_not_in_books, len(df), 'full: 제외')
        add(name, 'exact_duplicate', '일관성', '모든 열이 같은 중복 행', f.exact_duplicate, len(df), 'full: 제외')
        if name == '입하':
            before = df.date < df.isbn.map(pub)
            add(name, 'before_publication', '일관성', '출간일 이전 입하', before, len(df), '예약 입하일 수 있어 기록만 함')
        drop[name] = f
    # 반품이 그때까지의 누적 입하를 넘는지 (입하 기록 시작 전 공급분이 있으면 과대 탐지될 수 있다)
    flow = pd.concat([rcvd.assign(o=rcvd.qty, r=0), rtgd.assign(o=0, r=rtgd.qty)]).dropna(subset=['date']).sort_values(['date', 'r'], kind='stable')
    cum = flow.groupby('isbn')[['o', 'r']].cumsum()
    exceeds = ((cum.r > cum.o) & (flow.r > 0)).groupby(flow.index.where(flow.r > 0)).any() if len(flow) else pd.Series(dtype=bool)
    ex = pd.Series(False, index=rtgd.index)
    ex.loc[exceeds[exceeds].index.intersection(rtgd.index)] = True
    add('반품', 'return_exceeds_received', '일관성', '누적 반품 > 누적 입하 (입하 기록 이전 공급분 가능)', ex, len(rtgd), '기록만 함')

    months = pd.period_range(sales.month.min(), sales.month.max(), freq='M')
    missing = months.difference(pd.PeriodIndex(sales.month.unique()))
    add('판매', 'month_missing', '완전성', f'판매 월 누락 ({months[0]}~{months[-1]} 중)', pd.Series([True] * len(missing)), len(months), '해당 월 판매 변수 결측')
    add('판매', 'isbn_not_in_books', '일관성', '도서정보에 없는 ISBN의 판매 행', ~sales.isbn.isin(books.isbn), len(sales), 'full: 제외')
    add('판매', 'negative_sales', '일관성', '판매 수량 음수(반품 상계)', (sales[list(SALE_COLS.values())] < 0).any(axis=1), len(sales), '기록만 함')
    add('도서', 'pub_missing', '완전성', '출판일 누락', books.pub.isna(), len(books), '대상에서 제외 (두 조건 공통)')
    return rows, drop


def clean(books, rcvd, rtgd, sales, drop, full):
    r, t = rcvd[~drop['입하'].date_invalid], rtgd[~drop['반품'].date_invalid]
    if full:
        bad = lambda f: f.qty_nonpositive | f.isbn_invalid | f.isbn_not_in_books | f.exact_duplicate
        r, t = rcvd[~(drop['입하'].date_invalid | bad(drop['입하']))], rtgd[~(drop['반품'].date_invalid | bad(drop['반품']))]
        sales = sales[sales.isbn.isin(books.isbn)]
    return books[books.pub.notna()], r, t, sales


# ---------------------------------------------------------------- 변수와 라벨
def cutoffs(rcvd, sales):
    first = rcvd.date.min() + pd.Timedelta(days=ACTIVE_DAYS)
    last_sale = (sales.month.max() + 1).to_timestamp()  # 판매가 있는 마지막 달의 다음 달 1일까지 변수 계산 가능
    last_label = rcvd.date.max() - pd.Timedelta(days=HORIZON_DAYS)
    dates = pd.date_range(first.to_period('M').to_timestamp() + pd.offsets.MonthBegin(1), min(last_sale, last_label), freq='MS')
    if len(dates) < TEST_MONTHS + VAL_MONTHS + 6:
        raise ValueError(f'기준일 {len(dates)}개: 학습 6개 이상이 필요하다')
    s = [str(d.date()) for d in dates]
    return {'train': s[:-(TEST_MONTHS + VAL_MONTHS)], 'val': s[-(TEST_MONTHS + VAL_MONTHS):-TEST_MONTHS], 'test': s[-TEST_MONTHS:]}


def snapshot(books, rcvd, rtgd, sales, cutoff):
    days = lambda n: cutoff - pd.Timedelta(days=n)
    month = cutoff.to_period('M')
    r, t = rcvd[rcvd.date < cutoff], rtgd[rtgd.date < cutoff]
    s = sales[sales.month < month]
    r365, t365 = r[r.date >= days(365)], t[t.date >= days(365)]
    s12, s3, s1 = s[s.month >= month - 12], s[s.month >= month - 3], s[s.month == month - 1]
    active = set(r365.isbn) | set(s12[s12.total > 0].isbn)
    x = books[(books.pub < cutoff) & books.isbn.isin(active)][['isbn', 'pub', 'price', 'genre']].copy()
    by = lambda df, col='qty': x.isbn.map(df.groupby('isbn')[col].sum()).fillna(0)
    x['age_months'] = (cutoff - x.pub).dt.days / 30.4
    for n, part in ((30, r[r.date >= days(30)]), (90, r[r.date >= days(90)]), (365, r365)):
        x[f'rcvd_qty_{n}d'] = by(part)
        x[f'rcvd_n_{n}d'] = x.isbn.map(part.groupby('isbn').size()).fillna(0)
    x['days_since_rcvd'] = (cutoff - x.isbn.map(r.groupby('isbn').date.max())).dt.days
    x['rtgd_qty_90d'], x['rtgd_qty_365d'] = by(t[t.date >= days(90)]), by(t365)
    x['return_ratio_365d'] = (x.rtgd_qty_365d / x.rcvd_qty_365d).where(x.rcvd_qty_365d > 0)
    x['wtak_share_365d'] = x.isbn.map(r365[r365.buy == '위탁'].groupby('isbn').qty.sum()).fillna(0) / x.rcvd_qty_365d.where(x.rcvd_qty_365d > 0)
    x['sale_1m'], x['sale_3m'], x['sale_12m'] = by(s1, 'total'), by(s3, 'total'), by(s12, 'total')
    x['sale_store_3m'], x['sale_online_3m'] = by(s3, 'store'), by(s3, 'online')
    x['sale_months_12'] = x.isbn.map(s12[s12.total > 0].groupby('isbn').size()).fillna(0)
    x['sale_trend'] = (x.sale_1m / (x.sale_3m / 3)).where(x.sale_3m > 0)
    last_sale = x.isbn.map(s[s.total > 0].groupby('isbn').month.max())
    x['days_since_sale'] = [(cutoff - (m + 1).to_timestamp()).days if isinstance(m, pd.Period) else np.nan for m in last_sale]
    # 교보 재고 대리 변수: 최근 1년 입하 - 반품 - 판매
    x['net_flow_365d'] = x.rcvd_qty_365d - x.rtgd_qty_365d - x.sale_12m
    for g in sorted(books.genre.dropna().unique()):
        x[f'genre_{g}'] = x.genre.eq(g).astype(int)
    return x.drop(columns=['pub', 'genre'])


def reorder(rcvd, cutoff):
    return set(rcvd[(rcvd.date >= cutoff) & (rcvd.date < cutoff + pd.Timedelta(days=HORIZON_DAYS))].isbn)


def build(books, rcvd, rtgd, sales, cuts):
    frames = []
    for split, dates in cuts.items():
        for d in dates:
            c = pd.Timestamp(d)
            x = snapshot(books, rcvd, rtgd, sales, c)
            x['label'] = x.isbn.isin(reorder(rcvd, c)).astype(int)
            w = rtgd[(rtgd.date >= c) & (rtgd.date < c + pd.Timedelta(days=HORIZON_DAYS))]
            x['ret_qty'] = x.isbn.map(w.groupby('isbn').qty.sum()).fillna(0)
            x['ret_label'] = (x.ret_qty > 0).astype(int)
            frames.append(x.assign(split=split, cutoff=d))
    return pd.concat(frames, ignore_index=True)


FEATURES = None  # 첫 스냅샷에서 결정


def feature_cols(snaps):
    return [c for c in snaps.columns if c not in ('isbn', 'label', 'ret_label', 'ret_qty', 'split', 'cutoff')]


def rule_score(df):
    """최근 판매 규칙: 최근 3개월 판매량, 같으면 최근 입하가 가까운 도서 우선."""
    return (df.sale_3m - df.days_since_rcvd.fillna(9999) / 1e5).to_numpy(float)


def rfm_view(df):
    return pd.DataFrame({'recency_days': df.days_since_sale.fillna(9999), 'frequency': df.sale_months_12, 'monetary': df.sale_12m,
                         'label': df.label}, index=df.index)


# ---------------------------------------------------------------- 실험
def evaluate(name, snaps, truth, run_dir):
    cols = feature_cols(snaps)
    train, val = snaps[snaps.split == 'train'], snaps[snaps.split == 'val']
    test = snaps[snaps.split == 'test'].merge(truth, on=['isbn', 'cutoff']).sort_values(['cutoff', 'isbn']).reset_index(drop=True)
    test['label'] = test.pop('true_label')
    validation = {'rfm_rule': metrics(val.label, rule_score(val))}
    for m, model in make_models().items():
        validation[m] = metrics(val.label, model.fit(train[cols], train.label).predict_proba(val[cols])[:, 1])
    winner = max((m for m in validation if m != 'rfm_rule'), key=lambda m: validation[m]['precision20'])
    fit = pd.concat([train, val])
    scores = {'rfm_rule': rule_score(test)}
    for m, model in make_models().items():
        scores[m] = model.fit(fit[cols], fit.label).predict_proba(test[cols])[:, 1]
    # 평가는 기준일마다 상위 20%를 고르고 합쳐서 계산한다(매달 교보 대응 도서를 정하는 상황).
    y = test.label.to_numpy()
    results, gains = [], []
    per_month = lambda s, frac=CONTACT_FRACTION: monthly_pick(test, s, frac)
    for strategy, s in scores.items():
        pick = per_month(s)
        lo, hi = bootstrap_precision(y, s)
        results.append(dict(condition=name, strategy=strategy, selected_on_validation=strategy == winner, **pooled(y, s, pick),
                            precision20_ci_low=lo, precision20_ci_high=hi))
        for f in GAIN_FRACTIONS:
            p = per_month(s, f)
            gains.append(dict(condition=name, strategy=strategy, fraction=f, precision=float(y[p].mean()), recall=float(y[p].sum() / y.sum()) if y.sum() else 0))
    rnd = [pooled(y, r, per_month(r)) for r in (np.random.default_rng(k).random(len(y)) for k in range(30))]
    p = [r['precision20'] for r in rnd]
    results.append(dict(condition=name, strategy='random_30seeds', selected_on_validation=False, **{k: float(np.mean([r[k] for r in rnd])) for k in rnd[0]},
                        precision20_ci_low=min(p), precision20_ci_high=max(p)))
    chosen = scores[winner]
    test = test.assign(score=chosen, selected=per_month(chosen).astype(int))
    test[['cutoff', 'isbn', 'score', 'selected', 'label']].to_csv(run_dir / f'predictions_{name}.csv', index=False)
    detail = dict(condition=name, winner=winner, validation=validation, n_train=len(train), n_val=len(val), n_test=len(test), features=cols)
    return results, gains, detail, test


def monthly_pick(test, scores, fraction):
    pick = np.zeros(len(test), bool)
    for _, idx in test.groupby('cutoff').indices.items():
        k = max(1, math.ceil(len(idx) * fraction))
        pick[idx[np.argsort(-scores[idx], kind='stable')[:k]]] = True
    return pick


def pooled(y, scores, pick):
    base = float(y.mean())
    m = metrics(y, scores)  # AP·ROC-AUC는 전체 점수 기준
    prec = float(y[pick].mean())
    return dict(n=len(y), selected=int(pick.sum()), base_rate=base, precision20=prec, recall20=float(y[pick].sum() / y.sum()) if y.sum() else None,
                lift20=prec / base if base else None, average_precision=m['average_precision'], roc_auc=m['roc_auc'])


def book_segments(snaps, test):
    train = rfm_view(snaps[snaps.split == 'train'])
    last = test.cutoff.max()
    view = rfm_view(test[test.cutoff == last]).reset_index(drop=True)
    sel = test[test.cutoff == last].selected.to_numpy()
    rule = AccountRFM().fit(train)
    seg = rule.segment(view).str.replace('고객', '도서')
    rows = segment_table(view, seg, sel)
    return rows, last


def age_band(label):
    m = re.match(r'(\d+)', str(label))
    if not m:
        return '기타'
    a = int(m.group(1))
    return '10대 이하' if a < 20 else '20대' if a < 30 else '30대' if a < 40 else '40대' if a < 50 else '50대 이상'


def readers(src, books, test):
    """상품별 고객성향(교보 구매자 성별·연령 집계)으로 도서를 독자층별로 묶는다.
    수집 기간이 평가 구간과 겹치므로 예측 변수로 쓰지 않고 탐색 결과로만 보고한다."""
    path = src.parent / '고객성향'
    files = sorted(path.glob('*.json')) if path.exists() else []
    if not files:
        return None
    payload = json.loads(files[-1].read_text(encoding='utf8'))
    rows = []
    for r in payload['rows']:
        g, ages = r.get('gender', {}), {}
        for k, v in r.get('age_total', {}).items():
            ages[age_band(k)] = ages.get(age_band(k), 0) + v
        total, known = sum(g.values()), g.get('남자', 0) + g.get('여자', 0)
        known_age = {k: v for k, v in ages.items() if k != '기타'}
        top = max(known_age, key=known_age.get) if known_age else None
        female = g.get('여자', 0) / known if known else np.nan
        if known < 5 or top is None:
            kind = '표본 부족 (성별 확인 5권 미만)'
        else:
            kind = f"{top} {'여성' if female >= .6 else '남성' if female <= .4 else '남녀'} 중심"
        rows.append(dict(isbn=r['isbn'], copies=total, known_share=known / total if total else np.nan, female_share=female, reader_type=kind,
                         **{f'age_{k}': ages.get(k, 0) for k in ('10대 이하', '20대', '30대', '40대', '50대 이상', '기타')}))
    x = pd.DataFrame(rows).merge(books[['isbn', 'genre']], on='isbn', how='left')
    reorder_rate = test.groupby('isbn').label.mean()
    x['test_reorder_rate'] = x.isbn.map(reorder_rate)
    overall = {k: int(x[f'age_{k}'].sum()) for k in ('10대 이하', '20대', '30대', '40대', '50대 이상', '기타')}
    table = []
    for kind, part in x.groupby('reader_type'):
        table.append(dict(segment=kind, n=len(part), copies=int(part.copies.sum()), female_share=float(part.female_share.mean()),
                          known_share=float((part.copies * part.known_share).sum() / part.copies.sum()) if part.copies.sum() else None,
                          genres=', '.join(f'{g} {c}' for g, c in part.genre.value_counts().head(3).items()),
                          test_reorder_rate=float(part.test_reorder_rate.mean()) if part.test_reorder_rate.notna().any() else None))
    table.sort(key=lambda r: -r['copies'])
    return dict(period=payload.get('period'), titles=len(x), copies=int(x.copies.sum()), overall_age=overall,
                gender={k: int(sum(r.get('gender', {}).get(k, 0) for r in payload['rows'])) for k in ('남자', '여자', '기타')},
                segments=table)


def run(args):
    started = time.time()
    books, rcvd, rtgd, sales, hashes = load(args.src)
    qrows, drop = quality(books, rcvd, rtgd, sales)
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    run_dir = args.output / run_id
    run_dir.mkdir(parents=True)
    db = sqlite3.connect(args.output / 'research.sqlite')
    db.executescript(SCHEMA)
    db.execute('CREATE TABLE IF NOT EXISTS quality_checks_kyobo(run_id TEXT, tbl TEXT, check_name TEXT, dimension TEXT, rule TEXT, violations INTEGER, share REAL, action TEXT)')

    def save(stage, payload):
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str)
        db.execute('INSERT INTO artifacts VALUES(?,?,?)', (run_id, stage, text))
        (run_dir / f'{stage}.json').write_text(text, encoding='utf8')

    data = {n: clean(books, rcvd, rtgd, sales, drop, full) for n, full in (('minimal', False), ('full', True))}
    cuts = cutoffs(data['full'][1], data['full'][3])
    config = dict(run_id=run_id, store=report.STORE['name'], source=report.STORE['source'], files=hashes, horizon_days=HORIZON_DAYS,
                  contact_fraction=CONTACT_FRACTION, cutoffs=cuts, label=f"기준일 이후 30일 안에 {report.STORE['name']} {report.STORE['event']}(재주문) 1건 이상",
                  sales_months=[str(sales.month.min()), str(sales.month.max())],
                  agent_mode='규칙 기반 검증기만 구현. LLM 멀티 에이전트 미실행', standard='ISO/IEC 25012 참고 프로젝트 자체 점검. 적합성 인증 아님')
    db.execute('INSERT INTO runs VALUES(?,?,?,?)', (run_id, time.strftime('%Y-%m-%d %H:%M:%S'), json.dumps(hashes), json.dumps(config, ensure_ascii=False)))
    db.executemany('INSERT INTO quality_checks_kyobo VALUES(?,?,?,?,?,?,?,?)',
                   [(run_id, q['table'], q['check'], q['dimension'], q['rule'], q['violations'], q['share'], q['action']) for q in qrows])
    snaps = {n: build(*d, cuts) for n, d in data.items()}
    truth = snaps['full'].query("split == 'test'")[['isbn', 'cutoff', 'label']].rename(columns={'label': 'true_label'})
    base = [dict(condition=n, split=sp, cutoff=c, customers=len(p), base_rate=float(p.label.mean()))
            for n, s in snaps.items() for (sp, c), p in s.groupby(['split', 'cutoff'], sort=False)]
    summary = dict(books=len(books), rcvd_rows=len(rcvd), rtgd_rows=len(rtgd), sales_rows=len(sales), rcvd_qty=float(rcvd.qty.sum()),
                   rtgd_qty=float(rtgd.qty.sum()), return_reasons=rtgd.groupby('reason').qty.sum().sort_values(ascending=False).to_dict(),
                   buy_mix=rcvd.groupby('buy').qty.sum().to_dict())
    save('01_quality', dict(config=config, data=summary, quality=qrows, base_rates=base))

    results, gains, details = [], [], []
    for n in ('minimal', 'full'):
        r, g, d, test = evaluate(n, snaps[n], truth, run_dir)
        results += r; gains += g; details.append(d)
        if n == 'full':
            d['segments'], d['segment_cutoff'] = book_segments(snaps[n], test)
            d['readers'] = readers(args.src, data['full'][0], test)
    save('02_models', dict(results=results, gains=gains, details=details))
    save('04_crm_roi', crm_roi.analyze(args.src, data['full'][0], data['full'][1], data['full'][3], snaps['full'], feature_cols(snaps['full']), monthly_pick))
    rows = [(f"{r['condition']}/test/{r['strategy']}/{m}", run_id, r['condition'], 'test', r['strategy'], m, r[m], r['n'])
            for r in results for m in ('precision20', 'recall20', 'lift20', 'average_precision', 'roc_auc', 'base_rate')]
    db.executemany('INSERT INTO metrics VALUES(?,?,?,?,?,?,?,?)', rows)
    evidence = {r[0]: r[6] for r in rows}
    claims = [dict(id=f'claim-{i}', kind='numeric', evidence_id=k, value=v) for i, (k, v) in enumerate(evidence.items()) if k.endswith('precision20')]
    save('03_review', dict(mode=config['agent_mode'], claims=claims, findings=review(claims, evidence)))
    db.commit(); db.close()
    out = build_report(run_dir)
    print(f'{out}\n{time.time() - started:.0f}s')
    return run_dir


# ---------------------------------------------------------------- 보고서
def build_report(run_dir):
    q = json.loads((run_dir / '01_quality.json').read_text(encoding='utf8'))
    m = json.loads((run_dir / '02_models.json').read_text(encoding='utf8'))
    cfg, data = q['config'], q['data']
    S = html.escape(cfg.get('store', report.STORE['name']))
    E = '발주' if cfg.get('store') == '예스24' else '입하'
    report.STRATEGY = {**report.STRATEGY, 'rfm_rule': '최근 판매 규칙'}
    pct = report.pct
    full = [r for r in m['results'] if r['condition'] == 'full']
    det = {d['condition']: d for d in m['details']}
    cards = ''.join(f'<div class="card">{a}<b>{b}</b></div>' for a, b in [
        ('도서', f"{data['books']:,}종"), ('입하 행', f"{data['rcvd_rows']:,}"), ('반품 행', f"{data['rtgd_rows']:,}"),
        ('평가 도서×월', f"{full[0]['n']:,}"), ('평가 재주문율', pct(full[0]['base_rate'])), ('선정 (매월 20%)', f"{full[0]['selected']:,}")])
    bars = ''.join(
        f'<div class="barrow"><span>{report.STRATEGY[r["strategy"]]}{" ★" if r["selected_on_validation"] else ""}</span><div class="track">'
        f'<div class="bar" style="width:{r["precision20"] * 100:.1f}%;background:{report.COLORS[r["strategy"]]}"></div></div>'
        f'<b>{pct(r["precision20"])} [{pct(r["precision20_ci_low"])}–{pct(r["precision20_ci_high"])}]</b></div>' for r in full)
    rc = [('condition', '조건', None), ('label', '전략', None), ('precision20', 'Precision@20%', pct), ('recall20', 'Recall@20%', pct),
          ('lift20', 'Lift@20%', lambda v: f'{v:.2f}'), ('average_precision', 'PR-AUC', lambda v: f'{v:.3f}'), ('roc_auc', 'ROC-AUC', lambda v: f'{v:.3f}')]
    res = [dict(r, label=report.STRATEGY[r['strategy']]) for r in m['results']]
    seg = [dict(s, ci=f"{pct(s['ci_low'])}–{pct(s['ci_high'])}") for s in det['full']['segments']]
    sc = [('segment', '도서군', None), ('n', '도서 수', lambda v: f'{v:,}'), ('response_rate', '30일 재주문율', pct), ('ci', '95% 구간', None),
          ('target_share', '선정 비율', pct), ('median_recency', '마지막 판매 후(일)', lambda v: f'{v:.0f}'),
          ('median_frequency', '12개월 중 판매 월', lambda v: f'{v:.0f}'), ('median_monetary', '12개월 판매(권)', lambda v: f'{v:,.0f}')]
    qc = [('table', '자료', None), ('dimension', '관점', None), ('rule', '점검 규칙', None), ('violations', '위반', lambda v: f'{v:,}'),
          ('share', '비율', lambda v: pct(v, 2)), ('action', '처리', None)]
    bc = [('condition', '조건', None), ('split', '구간', None), ('cutoff', '기준일', None), ('customers', '도서 수', lambda v: f'{v:,}'), ('base_rate', '30일 재주문율', pct)]
    extra = run_dir / '04_crm_roi.json'
    extra_html = crm_roi.sections(json.loads(extra.read_text(encoding='utf8'))) if extra.exists() else ''
    rd = det['full'].get('readers')
    reader_html = '<section><h2>독자층 분석</h2><p class="note">고객성향 자료가 없다.</p></section>'
    if rd:
        known = rd['gender']['남자'] + rd['gender']['여자']
        ages = ' · '.join(f'{k} {v:,}' for k, v in rd['overall_age'].items())
        rc2 = [('segment', '독자층', None), ('n', '도서 수', lambda v: f'{v:,}'), ('copies', '판매(권)', lambda v: f'{v:,}'),
               ('known_share', '성별 확인 비율', pct), ('female_share', '여성 비율(확인분)', pct), ('genres', '주요 분야', None),
               ('test_reorder_rate', '평가 구간 재주문율', pct)]
        reader_html = (f'<section><h2>독자층 분석 ({S} 구매자 {rd["period"][0]} ~ {rd["period"][1]})</h2>'
                       f'<p class="note">도서 {rd["titles"]}종 · {rd["copies"]:,}권. 성별: 남자 {rd["gender"]["남자"]:,} / 여자 {rd["gender"]["여자"]:,} / 기타(미확인) {rd["gender"]["기타"]:,}. '
                       f'성별이 확인된 구매는 {pct(known / rd["copies"] if rd["copies"] else None)}이다. 연령(권): {ages}.<br>'
                       '독자층은 성별·연령이 확인된 구매가 5권 이상인 도서만 분류했다. 수집 기간이 평가 구간과 겹쳐 예측 변수로 쓰지 않았으며, 재주문율과의 관계는 탐색 결과다.</p>'
                       f'{report.table(rd["segments"], rc2)}</section>')
    reasons = ''.join(f'<li>{html.escape(k)}: {v:,.0f}권</li>' for k, v in data['return_reasons'].items())
    body = f'''<p class="tag">{html.escape(cfg.get("source", S))} · 판매 {cfg["sales_months"][0]} ~ {cfg["sales_months"][1]} · 실행 {cfg["run_id"]}</p>
<h1>도서 세분화와 {S} 재주문 예측을 통한<br>영업·재고 대응 도서 선정</h1>
<p>매월 1일 기준으로, 그 이전 {E}·반품·판매로 도서 변수를 만들고 다음 30일 안의 {S} 재주문({E})을 예측한다. 평가: {cfg["cutoffs"]["test"][0]} ~ {cfg["cutoffs"]["test"][-1]} ({len(cfg["cutoffs"]["test"])}개월).</p>
<div class="cards">{cards}</div>
<section><h2>매월 상위 20% 도서 중 실제로 재주문된 비율</h2><p class="note">품질 점검 적용 조건 · ★ 검증 구간에서 선택된 모델 · 대괄호: 재표집 95% 구간(무작위는 30회 최소–최대)</p>{bars}</section>
<section><h2>누적 이득 곡선 (Recall)</h2>{report.gains_svg([g for g in m["gains"] if g["condition"] == "full"])}</section>
<section><h2>전략·조건별 평가</h2>{report.table(res, rc)}</section>
<section><h2>도서 RFM 세분화 ({det["full"]["segment_cutoff"]} 기준)</h2><p class="note">R=마지막 판매 후 경과, F=최근 12개월 중 판매가 있던 달 수, M=12개월 판매량. 탐색 결과이며 원인으로 해석하지 않는다.</p>{report.table(seg, sc)}</section>
{reader_html}
{extra_html}
<section><h2>반품 사유 (전체 기간, 권)</h2><ul>{reasons}</ul><p class="note">매입구분별 입하: {", ".join(f"{k} {v:,.0f}권" for k, v in data["buy_mix"].items())}</p></section>
<section><h2>데이터 품질 점검</h2><p class="note">{html.escape(cfg["standard"])}</p>{report.table(q["quality"], qc)}</section>
<section><h2>기준일별 도서 수와 재주문율</h2>{report.table(q["base_rates"], bc)}</section>
<section><h2>한계</h2><ul><li>{html.escape(cfg["agent_mode"])}.</li><li>{S} 1곳의 자료이며 다른 서점·총판은 포함하지 않는다.</li>
<li>입하 기록은 2021-04부터라 그 이전 공급분을 알 수 없다.</li><li>같은 도서가 여러 기준일에 반복 등장해 관측이 독립이 아니다. 구간은 참고용이다.</li>
<li>독자 자료는 {S} 구매자 집계이며 성별·연령 미확인(기타) 비중이 크다. 예측에는 쓰지 않았다.</li><li>영업 활동 기록이 없어 인과효과·ROI는 추정하지 않는다.</li></ul></section>'''
    out = run_dir / 'index.html'
    out.write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{S} 재주문 분석</title><style>{report.CSS}</style>{body}</html>', encoding='utf8')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=Path, default=SRC)
    p.add_argument('--output', type=Path, default=ROOT.parent / '결과')
    run(p.parse_args())
