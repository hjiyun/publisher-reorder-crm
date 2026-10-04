"""오류 삽입 실험: 원자료에 알려진 오류를 일부러 넣고, 품질 점검(full)이 재주문 예측을 얼마나 지키는지 본다.

실행:  python error_injection.py                 교보·예스24 모두
       python error_injection.py --store 교보 --seeds 3
산출:  ../결과/<run_id>/index.html, results.json, runs.csv

설계
- 자료: 교보(재주문 = 입하), 예스24(재주문 = 발주). 거래처 장부 없이 두 서점 자료만 쓴다.
- 오류는 원본 행에 넣는다(입하·발주, 반품·반출, 월별 판매). 유형마다 같은 개수씩.
  · 점검 규칙이 잡도록 만든 오류: 수량 0, 완전 중복 행, ISBN 체크 숫자 오타
  · 점검 규칙에 없는 오류: 수량 단위 오류(×10), 날짜 1년 오기, 행 누락
- 품질 점검 규칙은 kyobo.quality()·clean()을 그대로 쓴다. 넣은 오류에 맞춰 규칙을 고치지 않는다(사전 고정).
- 기준일은 원자료로 고정한다. 평가 정답과 평가 도서는 원자료(full 정제)에서 만든다.
  오염된 자료에서 사라진 평가 도서는 점수 최하위로 둔다(모델이 고를 수 없음).
- 학습 라벨과 변수는 오염된 자료에서 다시 만든다. 오류가 성능을 깎는 경로가 이것이다.
- 조건: minimal(일자 해석 불가만 제외) / full(추가 품질 점검). 같은 오염 자료에 두 조건을 적용해 짝지어 비교한다.
"""
import argparse
import html
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
sys.path.insert(0, str(BASE / '교보' / '코드'))
sys.path.insert(0, str(BASE / '예스24' / '코드'))

import common  # noqa: E402
import kyobo  # noqa: E402
import yes24  # noqa: E402

RATES = [0.0, 0.05, 0.1, 0.2]
TYPE_RATE = 0.1  # 유형별 실험의 오류 비율
STRATEGIES = ['rfm_rule', 'logistic', 'boosting']
TYPES = {
    'qty_zero': ('수량 0', True),
    'duplicate': ('완전 중복 행', True),
    'isbn_typo': ('ISBN 체크 숫자 오타', True),
    'qty_x10': ('수량 단위 오류 (×10)', False),
    'date_shift': ('날짜 1년 오기', False),
    'row_missing': ('행 누락', False),
}  # 값: (이름, 규칙이 잡도록 의도한 유형인가)
TABLES = {'rcvd': ['qty_zero', 'duplicate', 'isbn_typo', 'qty_x10', 'date_shift', 'row_missing'],
          'rtgd': ['qty_zero', 'duplicate', 'isbn_typo', 'qty_x10', 'date_shift', 'row_missing'],
          'sales': ['duplicate', 'isbn_typo', 'qty_x10', 'date_shift', 'row_missing']}
CHANNELS = ['store', 'online', 'interpark', 'corp', 'total']
STORES = {'교보': dict(load=lambda: kyobo.load(kyobo.SRC), event='입하'),
          '예스24': dict(load=lambda: yes24.load(yes24.SRC), event='발주')}


# ---------------------------------------------------------------- 오류 삽입
def typo(isbn):
    return isbn[:-1] + str((int(isbn[-1]) + 1) % 10) if isbn[-1:].isdigit() else isbn + 'X'


def inject_table(df, name, rate, types, rng, eligible):
    """df의 rate 비율 행에 types 오류를 같은 개수씩 넣는다. (오염된 표, 삽입 기록)
    원래부터 점검에 걸리는 행은 고르지 않는다. 그런 행이 제외되면 넣은 오류를 잡은 것처럼 보이기 때문이다."""
    df = df.reset_index(drop=True).copy()
    pool = np.flatnonzero(np.asarray(eligible))
    n = min(int(round(len(df) * rate)), len(pool))
    if n == 0 or not types:
        return df, []
    rows = rng.choice(pool, n, replace=False)
    log, extra, missing = [], [], []
    for i, idx in enumerate(rows):
        kind = types[i % len(types)]
        if kind == 'qty_zero':
            df.loc[idx, 'qty'] = 0
        elif kind == 'duplicate':
            extra.append(df.loc[[idx]])
            log.append(dict(table=name, type=kind, row=len(df) + len(extra) - 1))  # 뒤에 붙는 사본이 오류 행
            continue
        elif kind == 'isbn_typo':
            df.loc[idx, 'isbn'] = typo(str(df.loc[idx, 'isbn']))
        elif kind == 'qty_x10':
            cols = [c for c in CHANNELS if c in df.columns] if name == 'sales' else ['qty']
            df.loc[idx, cols] = df.loc[idx, cols] * 10
        elif kind == 'date_shift':
            if name == 'sales':
                lo, hi, step, col = df.month.min(), df.month.max(), 12, 'month'
            else:
                lo, hi, step, col = df.date.min(), df.date.max(), pd.Timedelta(days=365), 'date'
            v, sign = df.loc[idx, col], rng.choice([-1, 1])
            new = v + sign * step
            if not lo <= new <= hi:
                new = v - sign * step
            df.loc[idx, col] = new  # 양쪽 다 범위 밖이면 범위 밖 날짜가 된다(드묾)
        elif kind == 'row_missing':
            missing.append(idx)
        log.append(dict(table=name, type=kind, row=int(idx)))
    if extra:
        df = pd.concat([df, *extra], ignore_index=True)
    df = df.drop(index=missing)
    return df, log


def inject(books, rcvd, rtgd, sales, rate, types, seed):
    rng = np.random.default_rng(seed)
    rcvd, rtgd, sales = (d.reset_index(drop=True) for d in (rcvd, rtgd, sales))
    _, drop = kyobo.quality(books, rcvd, rtgd, sales)
    ok = lambda f: ~f.any(axis=1)
    eligible = {'rcvd': ok(drop['입하']), 'rtgd': ok(drop['반품']), 'sales': sales.isbn.isin(books.isbn)}
    out, log = {}, []
    for name, df in (('rcvd', rcvd), ('rtgd', rtgd), ('sales', sales)):
        kinds = [t for t in TABLES[name] if t in types]
        out[name], entries = inject_table(df, name, rate, kinds, rng, eligible[name])
        log += entries
    if 'total' in out['sales']:
        out['sales']['total'] = out['sales'][['store', 'online', 'interpark', 'corp']].sum(axis=1)
    return books, out['rcvd'], out['rtgd'], out['sales'], log


def detection(log, data, full_data):
    """넣은 오류 행이 정제 후에도 남아 있는지로 탐지 여부를 정한다. 행 누락은 행 단위로 탐지할 수 없다."""
    kept = {('minimal', 'rcvd'): set(data[1].index), ('minimal', 'rtgd'): set(data[2].index), ('minimal', 'sales'): set(data[3].index),
            ('full', 'rcvd'): set(full_data[1].index), ('full', 'rtgd'): set(full_data[2].index), ('full', 'sales'): set(full_data[3].index)}
    rows = []
    for e in log:
        if e['type'] == 'row_missing':
            rows.append(dict(e, minimal=False, full=False))
        else:
            rows.append(dict(e, minimal=e['row'] not in kept['minimal', e['table']], full=e['row'] not in kept['full', e['table']]))
    return rows


# ---------------------------------------------------------------- 평가
def score(snaps, truth):
    """오염 자료로 학습하고, 원자료 기준 평가 도서·정답으로 기준일마다 상위 20%를 골라 정밀도를 잰다."""
    cols = kyobo.feature_cols(snaps)
    fit = snaps[snaps.split != 'test']
    test = truth.merge(snaps[snaps.split == 'test'].drop(columns='label'), on=['isbn', 'cutoff'], how='left', indicator=True)
    test = test.sort_values(['cutoff', 'isbn']).reset_index(drop=True)
    present = (test.pop('_merge') == 'both').to_numpy()
    y = test.true_label.to_numpy()
    scores = {'rfm_rule': np.full(len(test), -1e12)}
    if present.any():
        scores['rfm_rule'][present] = kyobo.rule_score(test[present])
    for m, model in common.make_models().items():
        s = np.full(len(test), -1e12)
        if present.any():
            s[present] = model.fit(fit[cols], fit.label).predict_proba(test.loc[present, cols])[:, 1]
        scores[m] = s
    out = {}
    for m, s in scores.items():
        pick = kyobo.monthly_pick(test, s, common.CONTACT_FRACTION)
        out[m] = (float(y[pick].mean()), common.metrics(y, s)['average_precision'])
    return out, int((~present).sum())


def label_noise(snaps, clean_snaps):
    """학습·검증 구간 라벨이 같은 조건의 원자료 라벨과 다른 비율과, 원자료 학습 대상 중 빠진 비율."""
    key = ['isbn', 'cutoff']
    a = clean_snaps[clean_snaps.split != 'test'][key + ['label']]
    b = snaps[snaps.split != 'test'][key + ['label']]
    m = a.merge(b, on=key, how='outer', suffixes=('_true', '_seen'), indicator=True)
    both = m[m._merge == 'both']
    return dict(flip=float((both.label_true != both.label_seen).mean()) if len(both) else None,
                lost=float((m._merge == 'left_only').sum() / len(a)), added=float((m._merge == 'right_only').sum() / len(a)))


def one_run(store, raw, cuts, truth, clean_snaps, rate, types, seed, design):
    books, rcvd, rtgd, sales, log = inject(*raw, rate, types, seed)
    _, drop = kyobo.quality(books, rcvd, rtgd, sales)
    data = {n: kyobo.clean(books, rcvd, rtgd, sales, drop, full) for n, full in (('minimal', False), ('full', True))}
    det = detection(log, data['minimal'], data['full'])
    rows = []
    for cond, d in data.items():
        snaps = kyobo.build(*d, cuts)
        prec, missing = score(snaps, truth)
        noise = label_noise(snaps, clean_snaps[cond])
        for m, (p, ap) in prec.items():
            rows.append(dict(store=store, design=design, rate=rate, types='+'.join(types), seed=seed, condition=cond,
                             strategy=m, precision20=p, average_precision=ap, missing_test=missing, **{f'label_{k}': v for k, v in noise.items()}))
    for d in det:
        d.update(store=store, design=design, rate=rate, seed=seed)
    return rows, det


def run_store(store, seeds, rates):
    raw = STORES[store]['load']()[:4]
    _, drop = kyobo.quality(*raw)
    clean_data = {n: kyobo.clean(*raw, drop, full) for n, full in (('minimal', False), ('full', True))}
    cuts = kyobo.cutoffs(clean_data['full'][1], clean_data['full'][3])
    clean_snaps = {n: kyobo.build(*d, cuts) for n, d in clean_data.items()}
    truth = clean_snaps['full'].query("split == 'test'")[['isbn', 'cutoff', 'label']].rename(columns={'label': 'true_label'})
    rows, det = [], []
    jobs = [('rate', r, list(TYPES), s) for r in rates for s in (range(1) if r == 0 else range(seeds))]
    jobs += [('type', TYPE_RATE, [t], s) for t in TYPES for s in range(seeds)]
    for i, (design, rate, types, seed) in enumerate(jobs):
        r, d = one_run(store, raw, cuts, truth, clean_snaps, rate, types, seed, design)
        rows += r; det += d
        print(f'\r{store} {i + 1}/{len(jobs)}', end='', flush=True)
    print()
    info = dict(store=store, event=STORES[store]['event'], cutoffs=cuts, test_rows=len(truth), base_rate=float(truth.true_label.mean()),
                rows={'rcvd': len(raw[1]), 'rtgd': len(raw[2]), 'sales': len(raw[3])})
    return rows, det, info


# ---------------------------------------------------------------- 요약
def summarize(runs, det):
    runs = pd.DataFrame(runs)
    det = pd.DataFrame(det)
    by_rate = []
    for (store, rate, strategy), g in runs[runs.design == 'rate'].groupby(['store', 'rate', 'strategy']):
        p = g.pivot(index='seed', columns='condition', values='precision20')
        diff = p.full - p.minimal
        by_rate.append(dict(store=store, rate=rate, strategy=strategy, seeds=len(p),
                            minimal=float(p.minimal.mean()), full=float(p.full.mean()),
                            minimal_sd=float(p.minimal.std(ddof=1)) if len(p) > 1 else 0.0, full_sd=float(p.full.std(ddof=1)) if len(p) > 1 else 0.0,
                            diff=float(diff.mean()), diff_min=float(diff.min()), diff_max=float(diff.max()), full_wins=int((diff > 0).sum()),
                            ties=int((diff == 0).sum()), ap_minimal=float(g[g.condition == 'minimal'].average_precision.mean()),
                            ap_full=float(g[g.condition == 'full'].average_precision.mean())))
    by_type = []
    clean = runs[(runs.design == 'rate') & (runs.rate == 0)].set_index(['store', 'condition', 'strategy']).precision20
    for (store, types, strategy), g in runs[runs.design == 'type'].groupby(['store', 'types', 'strategy']):
        p = g.pivot(index='seed', columns='condition', values='precision20')
        noise = g[g.strategy == strategy].groupby('condition')[['label_flip', 'label_lost', 'label_added']].mean()
        by_type.append(dict(store=store, type=types, strategy=strategy, minimal=float(p.minimal.mean()), full=float(p.full.mean()),
                            drop_minimal=float(p.minimal.mean() - clean[store, 'minimal', strategy]),
                            drop_full=float(p.full.mean() - clean[store, 'full', strategy]),
                            diff=float((p.full - p.minimal).mean()),
                            ap_minimal=float(g[g.condition == 'minimal'].average_precision.mean()), ap_full=float(g[g.condition == 'full'].average_precision.mean()),
                            flip_minimal=float(noise.loc['minimal', 'label_flip']), flip_full=float(noise.loc['full', 'label_flip'])))
    detect = []
    if len(det):
        for (store, table, kind), g in det[det.design == 'rate'].groupby(['store', 'table', 'type']):
            detect.append(dict(store=store, table=table, type=kind, injected=len(g), minimal=float(g.minimal.mean()), full=float(g.full.mean())))
    return dict(by_rate=by_rate, by_type=by_type, detection=detect)


# ---------------------------------------------------------------- 보고서
TABLE_NAME = {'rcvd': '재주문 이벤트 (교보 입하 / 예스24 발주)', 'rtgd': '반품·반출', 'sales': '월별 판매'}
STRAT = {'rfm_rule': '판매 규칙', 'logistic': '로지스틱', 'boosting': '부스팅'}
pp = lambda v: '–' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v * 100:+.1f}%p'
pc = lambda v: '–' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v * 100:.1f}%'


def render(summary, infos, config):
    t = lambda rows, cols: common.table(rows, cols)
    parts = [f'<h1>오류 삽입 실험 — 품질 점검이 재주문 예측을 지키는가</h1>',
             f'<p class="note">실행 {html.escape(config["run_id"])} · 시드 {config["seeds"]}개 · 비율 {", ".join(pc(r) for r in config["rates"])} · '
             f'유형별 실험 비율 {pc(TYPE_RATE)}. 거래처 장부 없이 교보·예스24 SCM 자료만 사용.</p>']
    parts.append('<section><h2>설계</h2><ul>'
                 '<li>원자료(재주문 이벤트·반품·월별 판매)의 일정 비율 행에 오류 6종을 같은 개수씩 넣는다.</li>'
                 '<li>품질 점검 규칙은 기존 kyobo.quality()·clean() 그대로 둔다. 넣은 오류에 맞춰 고치지 않았다.</li>'
                 '<li>평가 정답·평가 도서·기준일은 원자료에서 고정한다. 학습 변수와 라벨은 오염된 자료에서 다시 만든다.</li>'
                 '<li>minimal(일자 해석 불가만 제외)과 full(추가 품질 점검)을 같은 오염 자료에 적용해 시드별로 짝지어 비교한다.</li>'
                 '<li>지표: 기준일(매월 1일)마다 상위 20%를 고른 뒤 합친 Precision@20%. 보조로 평가 구간 전체 점수의 평균 정밀도(AP).</li></ul>'
                 + t([dict(type=TYPES[k][0], intended='예' if TYPES[k][1] else '아니오',
                           tables=', '.join(n for n, ts in (('이벤트', TABLES['rcvd']), ('반품', TABLES['rtgd']), ('판매', TABLES['sales'])) if k in ts)) for k in TYPES],
                     [('type', '오류 유형', None), ('intended', '규칙 대상으로 의도', None), ('tables', '넣은 표', None)])
                 + '</section>')
    for info in infos:
        parts.append(f'<section><h2>{html.escape(info["store"])} 자료</h2><p>재주문 = {info["event"]}. 평가 {len(info["cutoffs"]["test"])}개 기준일, '
                     f'도서×기준일 {info["test_rows"]:,}행, 재주문 비율 {pc(info["base_rate"])}. 원본 행: 이벤트 {info["rows"]["rcvd"]:,}, '
                     f'반품 {info["rows"]["rtgd"]:,}, 판매 {info["rows"]["sales"]:,}.</p></section>')
    det = summary['detection']
    parts.append('<section><h2>1. 오류 유형별 탐지율 (전체 오류 비율 실험 합산)</h2>'
                 '<p class="note">탐지 = 정제 후 그 오류 행이 빠졌는가. 행 누락은 행 단위 규칙으로 잡을 수 없어 0%.</p>'
                 + t([dict(d, store=d['store'], table=TABLE_NAME[d['table']], type=TYPES[d['type']][0]) for d in det],
                     [('store', '서점', None), ('table', '표', None), ('type', '오류', None), ('injected', '넣은 행', str),
                      ('minimal', 'minimal 탐지', pc), ('full', 'full 탐지', pc)]) + '</section>')
    parts.append('<section><h2>2. 오류 비율별 Precision@20%</h2><p class="note">차이 = full − minimal (시드별 짝 비교의 평균, [최소, 최대]). '
                 'full 우세 = full이 높았던 시드 수 / 동률.</p>'
                 + t([dict(r, strategy=STRAT[r['strategy']], range=f'[{pp(r["diff_min"])}, {pp(r["diff_max"])}]', wins=f'{r["full_wins"]}/{r["seeds"]} (동률 {r["ties"]})',
                           minimal_s=f'{pc(r["minimal"])} ± {r["minimal_sd"] * 100:.1f}', full_s=f'{pc(r["full"])} ± {r["full_sd"] * 100:.1f}') for r in summary['by_rate']],
                     [('store', '서점', None), ('rate', '오류 비율', pc), ('strategy', '전략', None), ('minimal_s', 'minimal', str), ('full_s', 'full', str),
                      ('diff', '차이', pp), ('range', '범위', str), ('wins', 'full 우세', str), ('ap_minimal', 'AP minimal', pc), ('ap_full', 'AP full', pc)]) + '</section>')
    parts.append(f'<section><h2>3. 오류 유형별 영향 (비율 {pc(TYPE_RATE)}, 한 유형만)</h2>'
                 '<p class="note">하락 = 오류 0% 실행 대비 변화. 라벨 뒤집힘 = 학습·검증 구간 라벨이 같은 조건의 원자료 라벨과 다른 비율.</p>'
                 + t([dict(r, type=TYPES[r['type']][0], strategy=STRAT[r['strategy']]) for r in summary['by_type']],
                     [('store', '서점', None), ('type', '오류', None), ('strategy', '전략', None), ('minimal', 'minimal', pc), ('full', 'full', pc),
                      ('drop_minimal', 'minimal 하락', pp), ('drop_full', 'full 하락', pp), ('diff', 'full − minimal', pp),
                      ('ap_minimal', 'AP minimal', pc), ('ap_full', 'AP full', pc),
                      ('flip_minimal', '라벨 뒤집힘 minimal', pc), ('flip_full', '라벨 뒤집힘 full', pc)]) + '</section>')
    parts.append('<section><h2>한계</h2><ul><li>오류는 무작위로 고르게 넣었다. 실제 오류는 특정 도서·시기에 몰릴 수 있다.</li>'
                 '<li>품질 점검 규칙은 기존 규칙 그대로다. 규칙에 없는 오류(단위·날짜·누락)는 다음 단계의 Agent 검토 대상이다.</li>'
                 '<li>교보·예스24 두 서점 자료만 쓴다. 출판사 거래처 장부는 포함하지 않는다.</li></ul></section>')
    return f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>오류 삽입 실험</title><style>{common.CSS}</style><body>{"".join(parts)}</body></html>'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--store', choices=[*STORES, 'all'], default='all')
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--rates', default=','.join(map(str, RATES)))
    p.add_argument('--output', type=Path, default=ROOT.parent / '결과')
    args = p.parse_args()
    rates = [float(r) for r in args.rates.split(',')]
    started = time.time()
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    out = args.output / run_id
    out.mkdir(parents=True)
    runs, det, infos = [], [], []
    for store in (STORES if args.store == 'all' else [args.store]):
        r, d, info = run_store(store, args.seeds, rates)
        runs += r; det += d; infos.append(info)
    summary = summarize(runs, det)
    config = dict(run_id=run_id, seeds=args.seeds, rates=rates, type_rate=TYPE_RATE, types={k: v[0] for k, v in TYPES.items()}, tables=TABLES)
    pd.DataFrame(runs).to_csv(out / 'runs.csv', index=False, encoding='utf-8-sig')
    (out / 'results.json').write_text(json.dumps(dict(config=config, stores=infos, **summary), ensure_ascii=False, indent=1, default=str), encoding='utf8')
    (out / 'index.html').write_text(render(summary, infos, config), encoding='utf8')
    print(f'{out / "index.html"}\n{time.time() - started:.0f}s')


if __name__ == '__main__':
    main()
