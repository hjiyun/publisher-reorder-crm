"""출판사 거래처(서점·총판) 분석: 거래처 세분화와 거래처×도서 90일 재주문 예측.

실행:  python publisher.py                          (실제 자료: ../거래처장부/ → ../결과/거래처/)
       python make_synthetic_publisher.py
       python publisher.py --data ../거래처장부/합성_SYNTHETIC --output ../결과/거래처_합성 --synthetic   (기능 검증용)
입력 파일과 열 이름은 ../문서/출판사_데이터요청서.docx 와 같다.
지표·모델·RFM·검증기는 common.py(Online Retail II로 검증한 절차에서 옮겨 온 것)에서 가져온다.

설계 요약
- 분석 단위: 기준일 시점의 거래처×도서 조합(기준일 이전 365일 안에 출고가 있었던 조합).
- 라벨: 기준일 이후 90일 안에 그 거래처가 그 도서를 다시 주문(출고)했는가.
- 기준일: 분기 초. 마지막 = 평가, 그 직전 = 검증, 나머지 = 학습. 라벨 구간이 다음 기준일을 넘지 않는다.
- 조건 minimal(실행에 필요한 최소 처리)과 full(추가 품질 점검)을 같은 평가 조합·같은 정답으로 비교한다.
"""
import argparse
import hashlib
import html
import json
import math
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import common as report
from common import (CONTACT_FRACTION, GAIN_FRACTIONS, RFM, SCHEMA, AccountRFM, bootstrap_precision, isbn13_valid, make_models, metrics,
                    review, segment_table)

ROOT = Path(__file__).resolve().parent
HORIZON_DAYS = 90
ACTIVE_DAYS = 365
KEYS = ['account', 'isbn']
FILES = {
    'transactions': ('출고반품', {'전표번호': 'doc_id', '일자': 'date', '거래처코드': 'account', '지점코드': 'branch', 'ISBN': 'isbn',
                                  '구분': 'kind', '수량': 'qty', '단가': 'unit_price', '금액': 'amount', '원전표번호': 'orig_doc'}),
    'accounts': ('거래처', {'거래처코드': 'account', '거래처유형': 'account_type', '지역': 'region', '거래시작일': 'since'}),
    'books': ('도서', {'ISBN': 'isbn', '분야': 'genre', '저자코드': 'author', '시리즈': 'series', '출간일': 'pub_date', '정가': 'list_price'}),
    'scm': ('서점판매', {'ISBN': 'isbn', '주차시작일': 'week', '판매수량': 'sold'}),  # 선택 파일
}
KIND = {'출고': 'out', '반품': 'return'}
NUMERIC = ['recency_days', 'pair_tenure_days', 'orders', 'qty', 'orders_90d', 'qty_90d', 'orders_365d', 'qty_365d',
           'return_qty', 'pair_return_rate', 'acct_orders_90d', 'acct_titles_365d', 'acct_return_rate',
           'book_qty_90d', 'book_accounts_90d', 'book_return_rate', 'book_age_days', 'list_price']


# ---------------------------------------------------------------- 적재
def read_table(data_dir, stem):
    for path in (data_dir / f'{stem}.csv', data_dir / f'{stem}.xlsx'):
        if not path.exists():
            continue
        raw = path.read_bytes()
        if path.suffix == '.xlsx':
            return pd.read_excel(path, dtype=str), hashlib.sha256(raw).hexdigest()
        for encoding in ('utf-8-sig', 'cp949'):  # 한글 엑셀에서 저장한 CSV는 cp949인 경우가 많다
            try:
                return pd.read_csv(path, dtype=str, encoding=encoding), hashlib.sha256(raw).hexdigest()
            except UnicodeDecodeError:
                continue
    return None, None


def clean_isbn(s):
    return s.str.replace('-', '', regex=False).str.strip().replace('', np.nan)


def load(data_dir):
    frames, hashes = {}, {}
    for key, (stem, cols) in FILES.items():
        df, sha = read_table(data_dir, stem)
        if df is None:
            if key == 'scm':
                frames[key] = None
                continue
            raise FileNotFoundError(data_dir / f'{stem}.csv')
        df.columns = df.columns.str.strip()
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f'{stem}: 필요한 열이 없음 {sorted(missing)}')
        frames[key], hashes[stem] = df[list(cols)].rename(columns=cols), sha

    tx = frames['transactions']
    for c in ('doc_id', 'account', 'branch', 'orig_doc', 'kind'):
        tx[c] = tx[c].str.strip().replace('', np.nan)
    tx['isbn'] = clean_isbn(tx.isbn)
    tx['date'] = pd.to_datetime(tx.date, errors='coerce')
    for c in ('qty', 'unit_price', 'amount'):
        tx[c] = pd.to_numeric(tx[c].str.replace(',', '', regex=False), errors='coerce')
    tx['kind'] = tx.kind.map(KIND)  # 출고/반품 이외 값은 결측 -> kind_invalid
    tx['value'] = tx.amount.abs().fillna(tx.qty.abs() * tx.unit_price)

    books = frames['books']
    books['isbn'] = clean_isbn(books.isbn)
    books['pub_date'] = pd.to_datetime(books.pub_date, errors='coerce')
    books['list_price'] = pd.to_numeric(books.list_price.str.replace(',', '', regex=False), errors='coerce')
    accounts = frames['accounts']
    accounts['account'] = accounts.account.str.strip()
    scm = frames['scm']
    if scm is not None:
        scm['isbn'] = clean_isbn(scm.isbn)
        scm['week'] = pd.to_datetime(scm.week, errors='coerce')
        scm['sold'] = pd.to_numeric(scm.sold.str.replace(',', '', regex=False), errors='coerce')
    return tx, books, accounts, scm, hashes


# ---------------------------------------------------------------- 품질 점검


QUALITY_RULES = {
    'account_missing': ('완전성', '거래처 코드 누락', '제외 (두 조건 공통)'),
    'isbn_missing': ('완전성', 'ISBN 누락', '제외 (두 조건 공통)'),
    'date_invalid': ('완전성', '일자 누락·해석 불가·미래 일자', '제외 (두 조건 공통)'),
    'kind_invalid': ('일관성', "구분이 '출고'/'반품'이 아님", '제외 (두 조건 공통)'),
    'qty_zero_or_missing': ('완전성', '수량 0 또는 누락', '제외 (두 조건 공통)'),
    'isbn_invalid': ('일관성', 'ISBN 형식·체크 숫자 오류', 'full: 제외'),
    'isbn_not_in_books': ('일관성', '도서정보에 없는 ISBN', 'full: 제외'),
    'account_not_in_master': ('일관성', '거래처 정보에 없는 거래처', 'full: 제외'),
    'sign_inconsistent': ('일관성', '출고인데 수량이 음수', 'full: 제외'),
    'return_unlinked': ('추적성', '원출고 전표와 연결되지 않는 반품', 'full: 제외'),
    'return_exceeds_shipped': ('일관성', '누적 반품이 누적 출고를 넘는 반품', 'full: 제외'),
    'exact_duplicate': ('일관성', '전표번호까지 같은 완전 중복 행', 'full: 제외'),
    'amount_mismatch': ('일관성', '금액 ≠ 수량×단가 (1% 또는 1원 초과)', '기록만 함'),
    'before_publication': ('일관성', '출간일 이전 출고', '예약 출고일 수 있어 기록만 함'),
}


def quality_flags(tx, books, accounts):
    out, ret = tx.kind.eq('out'), tx.kind.eq('return')
    pub = tx.isbn.map(books.drop_duplicates('isbn').set_index('isbn').pub_date)
    out_docs = set(tx.loc[out, 'doc_id'].dropna())
    # 날짜 순(같은 날은 출고 먼저)으로 누적 출고·반품을 비교한다.
    usable = tx.account.notna() & tx.isbn.notna() & tx.date.notna() & tx.kind.notna()
    q = tx.qty.abs()
    ordered = tx.assign(o=q.where(out, 0), r=q.where(ret, 0))[usable].sort_values(['date', 'kind'], kind='stable')
    cum = ordered.groupby(KEYS)[['o', 'r']].cumsum()
    exceeds = pd.Series(False, index=tx.index)
    exceeds.loc[cum.index] = (cum.r > cum.o).to_numpy() & ret.loc[cum.index].to_numpy()
    return pd.DataFrame({
        'account_missing': tx.account.isna(),
        'isbn_missing': tx.isbn.isna(),
        'date_invalid': tx.date.isna() | (tx.date > pd.Timestamp.now()),
        'kind_invalid': tx.kind.isna(),
        'qty_zero_or_missing': tx.qty.isna() | tx.qty.eq(0),
        'isbn_invalid': tx.isbn.notna() & ~isbn13_valid(tx.isbn),
        'isbn_not_in_books': tx.isbn.notna() & ~tx.isbn.isin(books.isbn),
        'account_not_in_master': tx.account.notna() & ~tx.account.isin(accounts.account),
        'sign_inconsistent': out & tx.qty.lt(0),
        'return_unlinked': ret & (tx.orig_doc.isna() | ~tx.orig_doc.isin(out_docs)),
        'return_exceeds_shipped': exceeds,
        'exact_duplicate': tx.duplicated(keep='first'),
        'amount_mismatch': tx.amount.notna() & tx.unit_price.notna() & tx.qty.notna()
                           & ((tx.amount.abs() - tx.qty.abs() * tx.unit_price).abs() > np.maximum(1, 0.01 * tx.amount.abs())),
        'before_publication': out & pub.notna() & (tx.date < pub),
    })


def quality_report(tx, flags):
    return [dict(check=name, dimension=d, rule=rule, violations=int(flags[name].sum()), share=float(flags[name].mean()), action=action)
            for name, (d, rule, action) in QUALITY_RULES.items()]


def clean(tx, flags, full):
    keep = ~(flags.account_missing | flags.isbn_missing | flags.date_invalid | flags.kind_invalid | flags.qty_zero_or_missing)
    if not full:
        return tx[keep]  # 수량 부호·반품 연결 등을 손대지 않은 그대로
    keep &= ~(flags.isbn_invalid | flags.isbn_not_in_books | flags.account_not_in_master | flags.sign_inconsistent
              | flags.return_unlinked | flags.return_exceeds_shipped | flags.exact_duplicate)
    return tx[keep].assign(qty=lambda t: t.qty.abs())


# ---------------------------------------------------------------- 기준일, 변수, 라벨
def make_cutoffs(tx):
    start, end = tx.date.min(), tx.date.max()
    dates = pd.date_range(start + pd.Timedelta(days=ACTIVE_DAYS), end - pd.Timedelta(days=HORIZON_DAYS), freq='QS')
    if len(dates) < 4:
        raise ValueError(f'기준일 {len(dates)}개: 학습 2개 이상 + 검증 + 평가가 필요하다 (기록 기간 약 2년 이상)')
    return {'train': [str(d.date()) for d in dates[:-2]], 'val': [str(dates[-2].date())], 'test': [str(dates[-1].date())]}


def snapshot(tx, books, accounts, scm, cutoff):
    """기준일 이전 기록만으로 만든 거래처×도서 변수. 기준일 이후 행은 읽지 않는다."""
    days = lambda n: cutoff - pd.Timedelta(days=n)
    past = tx[tx.date < cutoff]
    out, ret = past[past.kind == 'out'], past[past.kind == 'return']
    out365, out90 = out[out.date >= days(365)], out[out.date >= days(90)]

    x = out.groupby(KEYS).agg(first=('date', 'min'), last=('date', 'max'), orders=('qty', 'size'), qty=('qty', 'sum'))
    x = x[x['last'] >= days(ACTIVE_DAYS)].reset_index()
    x['recency_days'] = (cutoff - x['last']).dt.days
    x['pair_tenure_days'] = (cutoff - x['first']).dt.days
    for n, part in ((90, out90), (365, out365)):
        agg = part.groupby(KEYS).agg(**{f'orders_{n}d': ('qty', 'size'), f'qty_{n}d': ('qty', 'sum')}).reset_index()
        x = x.merge(agg, on=KEYS, how='left')
    x = x.merge(ret.groupby(KEYS).qty.sum().rename('return_qty').reset_index(), on=KEYS, how='left')
    x[['orders_90d', 'qty_90d', 'return_qty']] = x[['orders_90d', 'qty_90d', 'return_qty']].fillna(0)
    x['pair_return_rate'] = (x.return_qty / x.qty).where(x.qty > 0)

    acct = pd.DataFrame({'acct_orders_90d': out90.groupby('account').size(),
                         'acct_titles_365d': out365.groupby('account').isbn.nunique(),
                         'acct_return_rate': ret.groupby('account').qty.sum() / out.groupby('account').qty.sum()})
    book = pd.DataFrame({'book_qty_90d': out90.groupby('isbn').qty.sum(),
                         'book_accounts_90d': out90.groupby('isbn').account.nunique(),
                         'book_return_rate': ret.groupby('isbn').qty.sum() / out.groupby('isbn').qty.sum()})
    x = x.merge(acct, left_on='account', right_index=True, how='left').merge(book, left_on='isbn', right_index=True, how='left')
    x[['acct_orders_90d', 'book_qty_90d', 'book_accounts_90d']] = x[['acct_orders_90d', 'book_qty_90d', 'book_accounts_90d']].fillna(0)

    meta = books.drop_duplicates('isbn').set_index('isbn')
    x['book_age_days'] = (cutoff - x.isbn.map(meta.pub_date)).dt.days
    x['list_price'] = x.isbn.map(meta.list_price)
    if scm is not None:  # 주간 판매는 주 전체가 기준일 전에 끝난 주만 쓴다
        weeks = scm[(scm.week <= days(7)) & (scm.week > days(35))]
        x['scm_sold_28d'] = x.isbn.map(weeks.groupby('isbn').sold.sum()).fillna(0)
    account_type = x.account.map(accounts.drop_duplicates('account').set_index('account').account_type)
    for t in sorted(accounts.account_type.dropna().unique()):
        x[f'type_{t}'] = account_type.eq(t).astype(int)
    genre = x.isbn.map(meta.genre)
    for g in sorted(books.genre.dropna().unique()):
        x[f'genre_{g}'] = genre.eq(g).astype(int)
    return x.drop(columns=['first', 'last'])


def reorders(tx, cutoff):
    w = tx[(tx.kind == 'out') & (tx.date >= cutoff) & (tx.date < cutoff + pd.Timedelta(days=HORIZON_DAYS))]
    return set(zip(w.account, w.isbn))


def build_snapshots(tx, books, accounts, scm, cutoffs):
    frames = []
    for split, dates in cutoffs.items():
        for d in dates:
            cutoff = pd.Timestamp(d)
            x = snapshot(tx, books, accounts, scm, cutoff)
            positive = reorders(tx, cutoff)
            x['label'] = [int(k in positive) for k in zip(x.account, x.isbn)]
            frames.append(x.assign(split=split, cutoff=d))
    return pd.concat(frames, ignore_index=True)


def features(snaps):
    extra = [c for c in snaps.columns if c.startswith(('scm_', 'type_', 'genre_'))]
    return NUMERIC + extra


def rf_view(df):
    """조합 단위 RFM: 최근 주문일, 365일 주문 횟수, 365일 출고 수량."""
    return pd.DataFrame({'recency_days': df.recency_days, 'frequency': df.orders_365d, 'monetary': df.qty_365d}, index=df.index)




def account_snapshot(tx, cutoff):
    days = lambda n: cutoff - pd.Timedelta(days=n)
    past = tx[tx.date < cutoff]
    out, ret = past[past.kind == 'out'], past[past.kind == 'return']
    out365, ret365 = out[out.date >= days(365)], ret[ret.date >= days(365)]
    a = pd.DataFrame({'last': out.groupby('account').date.max(), 'frequency': out365.groupby('account').size(),
                      'monetary': out365.groupby('account').value.sum().sub(ret365.groupby('account').value.sum(), fill_value=0),
                      'return_rate': ret365.groupby('account').qty.sum() / out365.groupby('account').qty.sum()})
    a = a[a.frequency > 0].copy()
    a['recency_days'] = (cutoff - a['last']).dt.days
    window = tx[(tx.kind == 'out') & (tx.date >= cutoff) & (tx.date < cutoff + pd.Timedelta(days=HORIZON_DAYS))]
    a['label'] = a.index.isin(window.account).astype(int)
    return a.drop(columns='last')


# ---------------------------------------------------------------- 실험
def evaluate(name, snaps, truth, run_dir):
    cols = features(snaps)
    train, val = snaps[snaps.split == 'train'], snaps[snaps.split == 'val']
    test = snaps[snaps.split == 'test'].merge(truth, on=KEYS).sort_values(KEYS).reset_index(drop=True)
    test['label'] = test.pop('true_label')
    rule = RFM().fit(rf_view(train))

    validation = {'rfm_rule': metrics(val.label, rule.predict(rf_view(val)))}
    for model_name, model in make_models().items():
        validation[model_name] = metrics(val.label, model.fit(train[cols], train.label).predict_proba(val[cols])[:, 1])
    winner = max((m for m in validation if m != 'rfm_rule'), key=lambda m: validation[m]['precision20'])

    fit_data = pd.concat([train, val])
    scores = {'rfm_rule': rule.predict(rf_view(test))}
    for model_name, model in make_models().items():
        scores[model_name] = model.fit(fit_data[cols], fit_data.label).predict_proba(test[cols])[:, 1]

    y = test.label.to_numpy()
    results, gains = [], []
    for strategy, s in scores.items():
        lo, hi = bootstrap_precision(y, s)
        results.append(dict(condition=name, strategy=strategy, selected_on_validation=strategy == winner, **metrics(y, s),
                            precision20_ci_low=lo, precision20_ci_high=hi))
        gains += [dict(condition=name, strategy=strategy, fraction=f, precision=metrics(y, s, f)['precision20'],
                       recall=metrics(y, s, f)['recall20']) for f in GAIN_FRACTIONS]
    randoms = [metrics(y, np.random.default_rng(seed).random(len(y))) for seed in range(30)]
    p = [r['precision20'] for r in randoms]
    results.append(dict(condition=name, strategy='random_30seeds', selected_on_validation=False,
                        **{k: float(np.mean([r[k] for r in randoms])) for k in randoms[0]},
                        precision20_ci_low=float(min(p)), precision20_ci_high=float(max(p))))

    chosen = scores[winner]
    selected = np.zeros(len(test), int)
    selected[np.argsort(-chosen, kind='stable')[:math.ceil(CONTACT_FRACTION * len(test))]] = 1
    test = test.assign(score=chosen, selected=selected)
    test[KEYS + ['score', 'selected', 'label']].to_csv(run_dir / f'predictions_{name}.csv', index=False)
    detail = dict(condition=name, validation=validation, winner=winner, features=cols,
                  n_train=len(train), n_val=len(val), n_test=len(test), brier=float(np.mean((chosen - y) ** 2)))
    return results, gains, detail, test


def account_segments(tx, cutoffs, selected_pairs):
    train = pd.concat([account_snapshot(tx, pd.Timestamp(d)) for d in cutoffs['train']])
    test = account_snapshot(tx, pd.Timestamp(cutoffs['test'][0]))
    rule = AccountRFM().fit(train)
    segments = rule.segment(test)
    share = selected_pairs.groupby('account').selected.mean().reindex(test.index).fillna(0).to_numpy()
    rows = segment_table(test, segments, share)
    total = test.monetary.clip(lower=0).sum()
    for r in rows:
        part = test[segments == r['segment']]
        r['median_return_rate'] = float(part.return_rate.median())
        r['value_share'] = float(part.monetary.clip(lower=0).sum() / total) if total else None
    return rows


def run(args):
    started = time.time()
    tx, books, accounts, scm, hashes = load(args.data)
    flags = quality_flags(tx, books, accounts)
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    run_dir = args.output / run_id
    run_dir.mkdir(parents=True)
    db = sqlite3.connect(args.output / 'research.sqlite')
    db.executescript(SCHEMA)

    def save(stage, payload):
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str)
        db.execute('INSERT INTO artifacts VALUES(?,?,?)', (run_id, stage, text))
        (run_dir / f'{stage}.json').write_text(text, encoding='utf8')
        return payload

    data = {name: clean(tx, flags, full) for name, full in (('minimal', False), ('full', True))}
    cutoffs = make_cutoffs(data['full'])
    config = dict(run_id=run_id, mode='SYNTHETIC TEST ONLY' if args.synthetic else 'PUBLISHER DATA', files=hashes,
                  horizon_days=HORIZON_DAYS, active_days=ACTIVE_DAYS, contact_fraction=CONTACT_FRACTION, cutoffs=cutoffs,
                  period=[str(tx.date.min()), str(tx.date.max())], scm_used=scm is not None,
                  label='기준일 이후 90일 안에 같은 거래처가 같은 도서를 다시 주문(출고)',
                  agent_mode='규칙 기반 검증기만 구현. LLM 멀티 에이전트 미실행',
                  standard='ISO/IEC 25012 참고 프로젝트 자체 점검. 적합성 인증 아님')
    db.execute('INSERT INTO runs VALUES(?,?,?,?)', (run_id, time.strftime('%Y-%m-%d %H:%M:%S'), json.dumps(hashes), json.dumps(config, ensure_ascii=False)))
    quality = quality_report(tx, flags)
    db.executemany('INSERT INTO quality_checks VALUES(?,?,?,?,?,?,?)',
                   [(run_id, q['check'], q['dimension'], q['rule'], q['violations'], q['share'], q['action']) for q in quality])

    snaps = {name: build_snapshots(t, books, accounts, scm, cutoffs) for name, t in data.items()}
    truth = snaps['full'].query("split == 'test'")[KEYS + ['label']].rename(columns={'label': 'true_label'})
    base_rates = [dict(condition=name, split=split, cutoff=cutoff, customers=len(part), base_rate=float(part.label.mean()))
                  for name, s in snaps.items() for (split, cutoff), part in s.groupby(['split', 'cutoff'], sort=False)]
    summary = dict(rows_total=len(tx), accounts=int(tx.account.nunique()), titles=int(tx.isbn.nunique()),
                   rows_by_condition={k: len(v) for k, v in data.items()})
    save('01_quality', dict(config=config, data=summary, quality=quality, base_rates=base_rates))

    results, gains, details = [], [], []
    for name in ('minimal', 'full'):
        r, g, d, test = evaluate(name, snaps[name], truth, run_dir)
        results += r; gains += g; details.append(d)
        if name == 'full':
            d['segments'] = account_segments(data['full'], cutoffs, test)
    save('02_models', dict(results=results, gains=gains, details=details))

    rows = [(f"{r['condition']}/test/{r['strategy']}/{m}", run_id, r['condition'], 'test', r['strategy'], m, r[m], r['n'])
            for r in results for m in ('precision20', 'recall20', 'lift20', 'average_precision', 'roc_auc', 'base_rate')]
    rows += [(f"full/test/accounts/{s['segment']}/response_rate", run_id, 'full', 'test', s['segment'], 'response_rate', s['response_rate'], s['n'])
             for s in details[1]['segments']]
    db.executemany('INSERT INTO metrics VALUES(?,?,?,?,?,?,?,?)', rows)
    evidence = {r[0]: r[6] for r in rows}
    claims = [dict(id=f'claim-{i}', kind='numeric', evidence_id=k, value=v) for i, (k, v) in enumerate(evidence.items()) if k.endswith('precision20')]
    save('03_review', dict(mode=config['agent_mode'], claims=claims, findings=review(claims, evidence)))
    db.commit(); db.close()
    index = build_report(run_dir)
    print(f'{index}\n{time.time() - started:.0f}s')
    return run_dir


# ---------------------------------------------------------------- 보고서
def build_report(run_dir):
    q = json.loads((run_dir / '01_quality.json').read_text(encoding='utf8'))
    m = json.loads((run_dir / '02_models.json').read_text(encoding='utf8'))
    cfg, data = q['config'], q['data']
    full = [r for r in m['results'] if r['condition'] == 'full']
    detail = {d['condition']: d for d in m['details']}
    pct = report.pct
    banner = ('<section style="background:#fde8e8"><b>합성 데이터 — 파이프라인 기능 검증용이며 연구 결과가 아닙니다.</b></section>'
              if cfg['mode'].startswith('SYNTHETIC') else '')
    cards = ''.join(f'<div class="card">{label}<b>{value}</b></div>' for label, value in [
        ('출고·반품 행', f"{data['rows_total']:,}"), ('거래처', f"{data['accounts']:,}"), ('도서', f"{data['titles']:,}"),
        ('평가 조합', f"{full[0]['n']:,}"), ('평가 재주문율', pct(full[0]['base_rate'])), ('집중 관리 (20%)', f"{full[0]['selected']:,}")])
    bars = ''.join(
        f'<div class="barrow"><span>{report.STRATEGY[r["strategy"]]}{" ★" if r["selected_on_validation"] else ""}</span><div class="track">'
        f'<div class="bar" style="width:{r["precision20"] * 100:.1f}%;background:{report.COLORS[r["strategy"]]}"></div></div>'
        f'<b>{pct(r["precision20"])} [{pct(r["precision20_ci_low"])}–{pct(r["precision20_ci_high"])}]</b></div>' for r in full)
    result_cols = [('condition', '조건', None), ('strategy_label', '전략', None), ('precision20', 'Precision@20%', pct), ('recall20', 'Recall@20%', pct),
                   ('lift20', 'Lift@20%', lambda v: f'{v:.2f}'), ('average_precision', 'PR-AUC', lambda v: f'{v:.3f}'), ('roc_auc', 'ROC-AUC', lambda v: f'{v:.3f}')]
    results = [dict(r, strategy_label=report.STRATEGY[r['strategy']]) for r in m['results']]
    seg_cols = [('segment', '거래처군', None), ('n', '거래처 수', lambda v: f'{v:,}'), ('value_share', '순출고 금액 비중', pct),
                ('response_rate', '90일 주문률', pct), ('ci', '95% 구간', None), ('median_return_rate', '반품률(중앙값)', pct),
                ('target_share', '선정된 조합 비율', pct), ('median_recency', '최근 주문(일)', lambda v: f'{v:.0f}'),
                ('median_frequency', '365일 주문 수', lambda v: f'{v:.0f}')]
    segs = [dict(s, ci=f"{pct(s['ci_low'])}–{pct(s['ci_high'])}") for s in detail['full']['segments']]
    quality_cols = [('dimension', '관점', None), ('rule', '점검 규칙', None), ('violations', '위반 행', lambda v: f'{v:,}'),
                    ('share', '비율', lambda v: pct(v, 2)), ('action', '처리', None)]
    base_cols = [('condition', '조건', None), ('split', '구간', None), ('cutoff', '기준일', None), ('customers', '조합 수', lambda v: f'{v:,}'),
                 ('base_rate', '90일 재주문율', pct)]
    body = f'''{banner}<p class="tag">출판사 거래처 분석 · {cfg["period"][0][:10]} ~ {cfg["period"][1][:10]} · 실행 {cfg["run_id"]}</p>
<h1>거래처 세분화와 재주문 예측을 통한<br>영업·마케팅 타깃 선정</h1>
<p>기준일 이전 출고·반품으로 거래처×도서 변수를 만들고, 기준일 이후 {cfg["horizon_days"]}일 안의 재주문을 예측한다. 평가 기준일 {cfg["cutoffs"]["test"][0]}.</p>
<div class="cards">{cards}</div>
<section><h2>선정한 상위 20% 조합 중 실제 재주문 비율</h2><p class="note">품질 점검 적용 조건 · ★ 검증 구간에서 선택된 모델</p>{bars}</section>
<section><h2>누적 이득 곡선 (Recall)</h2>{report.gains_svg([g for g in m["gains"] if g["condition"] == "full"])}</section>
<section><h2>전략·조건별 평가 결과</h2>{report.table(results, result_cols)}</section>
<section><h2>거래처 RFM 세분화 (평가 기준일)</h2><p class="note">탐색 결과이며 원인으로 해석하지 않는다.</p>{report.table(segs, seg_cols)}</section>
<section><h2>데이터 품질 점검</h2><p class="note">{html.escape(cfg["standard"])} · 사용 행: minimal {data["rows_by_condition"]["minimal"]:,} / full {data["rows_by_condition"]["full"]:,}</p>{report.table(q["quality"], quality_cols)}</section>
<section><h2>기준일별 조합 수와 재주문율</h2>{report.table(q["base_rates"], base_cols)}</section>
<section><h2>한계</h2><ul><li>{html.escape(cfg["agent_mode"])}.</li><li>출고는 서점의 주문이며 독자 판매와 같지 않다.</li>
<li>같은 거래처·도서가 여러 기준일에 반복 등장해 관측이 독립이 아니다.</li><li>영업 활동 기록과 비교집단이 없어 인과효과·ROI는 추정하지 않는다.</li></ul></section>'''
    out = run_dir / 'index.html'
    out.write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                   f'<title>출판사 거래처 분석 결과</title><style>{report.CSS}</style>{body}</html>', encoding='utf8')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT.parent / '거래처장부')
    p.add_argument('--output', type=Path, default=ROOT.parent / '결과' / '거래처')
    p.add_argument('--synthetic', action='store_true')
    run(p.parse_args())
