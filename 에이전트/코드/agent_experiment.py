"""A/B/C/D 실험: 품질 점검 유무 × LLM 에이전트 검토 유무를 오류 삽입 자료에서 비교한다.

실행:  python agent_experiment.py --pilot              교보 1건(오류 20%, 시드 0)으로 비용·동작 확인
       python agent_experiment.py                      교보·예스24, 오류 0·10·20%, 시드 2개
       python agent_experiment.py --no-llm             A·B·H만 (API 호출 없음)
산출:  ../결과/<run_id>/index.html, results.json, runs.csv · 에이전트 출력은 ../결과/agent.sqlite (같은 입력이면 다시 부르지 않음)

조건
- A: minimal(일자 해석 불가만 제외)            - B: full(기존 품질 점검)
- C: minimal 뒤 에이전트 검토                   - D: full 뒤 에이전트 검토
- H: full 뒤 고정 휴리스틱(조치 목록 중 일부를 정해진 값으로 일괄 적용, LLM 없음). 에이전트의 판단이 조치 목록 자체보다 나은지 보는 대조군
평가는 오류실험과 같다: 기준일·평가 도서·정답은 원자료, 학습은 오염 자료. 탐지 = 넣은 오류 행을 지웠거나 원래 값으로 되돌렸는가.

중간 기록 (실행 폴더 하나만 올리면 되도록 모은다. 원자료 행은 들어가지 않는다)
- run.log            단계별 진행 기록
- config.json        실행 설정 (시작 시 저장)
- agents/<사례>.json 에이전트 실행마다 즉시 저장: 분석가 A·B 독립 답변, 조정 결과, 검토 판정, 근거(도구 집계 결과), 최종 조치
- results.json, runs.csv, index.html  사례가 끝날 때마다 다시 써서 중간 결과를 남긴다
- agent.sqlite       끝날 때 에이전트 DB 사본, manifest.json  파일 목록과 SHA-256
"""
import argparse
import hashlib
import html
import json
import shutil
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / '오류실험' / '코드'))

import agents as ag  # noqa: E402
import common  # noqa: E402
import error_injection as ei  # noqa: E402
import kyobo  # noqa: E402

CONDITIONS = {'A': '점검 없음', 'B': '품질 점검', 'C': '에이전트만', 'D': '품질 점검 + 에이전트', 'H': '품질 점검 + 고정 휴리스틱'}
HEURISTIC = [dict(action='drop_key_duplicates', table='sales', keys=['isbn', 'month']),
             dict(action='rescale_qty_outliers', table='rcvd', threshold=8, require_multiple_of_10=True),
             dict(action='rescale_qty_outliers', table='rtgd', threshold=8, require_multiple_of_10=True),
             dict(action='rescale_qty_outliers', table='sales', threshold=8, require_multiple_of_10=True),
             dict(action='repair_dates_from_doc', table='rcvd', max_days=180),
             dict(action='repair_dates_from_doc', table='rtgd', max_days=180)]


def as_tables(d):
    return {'rcvd': d[1], 'rtgd': d[2], 'sales': d[3]}


def outcome(log, before, after):
    """넣은 오류의 처리 결과(지움·되돌림·남음)와, 오류가 아닌 행을 건드린 수(부수 피해).
    before는 조치 직전 표(minimal 또는 full 정제 결과)다. 규칙 정제가 지운 행은 부수 피해에 넣지 않는다."""
    rows, injected = [], {(e['table'], e['row']) for e in log}
    for e in log:
        t = after[e['table']]
        if e['type'] == 'row_missing':
            state = 'missing'
        elif e['row'] not in t.index:
            state = 'removed'
        elif e['type'] == 'qty_x10' and abs(float(t.loc[e['row'], 'total' if e['table'] == 'sales' else 'qty']) - e['orig']) < 1e-9:
            state = 'restored'
        elif e['type'] == 'date_shift' and str(t.loc[e['row'], 'month' if e['table'] == 'sales' else 'date']) == e['orig']:
            state = 'restored'
        else:
            state = 'kept'
        rows.append(dict(table=e['table'], type=e['type'], state=state))
    collateral = 0
    for n in ag.TABLES:
        b, a = before[n], after[n]
        clean = [i for i in b.index if (n, i) not in injected]
        gone = len(set(clean) - set(a.index))
        keep = a.index.intersection(clean)
        col = 'total' if n == 'sales' else 'qty'
        moved = (b.loc[keep, col] != a.loc[keep, col])
        if n != 'sales':
            moved |= b.loc[keep, 'date'] != a.loc[keep, 'date']
        collateral += gone + int(moved.sum())
    return rows, collateral


def one(store, raw, cuts, truth, clean_snaps, rate, seed, agents, run_id, out=None, log=print):
    books, rcvd, rtgd, sales, log = ei.inject(*raw, rate, list(ei.TYPES), seed)
    _, drop = kyobo.quality(books, rcvd, rtgd, sales)
    base = {c: as_tables(kyobo.clean(books, rcvd, rtgd, sales, drop, full)) for c, full in (('minimal', False), ('full', True))}
    plans = {'A': ('minimal', None), 'B': ('full', None), 'H': ('full', 'heuristic')}
    if agents is not None:
        plans.update(C=('minimal', 'agent'), D=('full', 'agent'))
    runs, det, reviews = [], [], {}
    for cond, (start, how) in plans.items():
        tables = base[start]
        if how == 'heuristic':
            tables, _ = ag.apply_actions(books, tables, HEURISTIC)
        elif how == 'agent':
            context = f'{store} 자료 ({"기존 품질 점검을 거친 뒤" if start == "full" else "품질 점검 없이"})'
            log(f'{store} {rate:.0%} s{seed} {cond}: 에이전트 검토 시작')
            rv = ag.review(agents, books, tables, context, run_id)
            tables, applied = ag.apply_actions(books, tables, rv['actions'])
            if out is not None:
                case = f'{store}_{int(rate * 100):02d}_s{seed}_{cond}'
                (out / 'agents').mkdir(exist_ok=True)
                (out / 'agents' / f'{case}.json').write_text(json.dumps(dict(case=case, context=context, **rv, applied=applied), ensure_ascii=False, indent=1, default=str), encoding='utf8')
            log(f'{store} {rate:.0%} s{seed} {cond}: 조치 {len(rv["actions"])}개 승인, 근거 없는 조치 {rv["unsupported_dropped"]}개 제외, '
                f'비용 ${rv["cost_usd"]:.2f} · ' + ', '.join(f'{m["role"]} {m["seconds"]}s' for m in rv['meta']))
            reviews[cond] = dict(actions=rv['actions'], applied=applied, unsupported_dropped=rv['unsupported_dropped'], meta=rv['meta'],
                                 cost_usd=rv['cost_usd'], findings_A=rv['analyst_A']['findings'], findings_B=rv['analyst_B']['findings'],
                                 moderator_dropped=rv['moderator']['dropped'], unresolved=rv['moderator']['unresolved'], verdicts=rv['reviewer']['verdicts'])
        states, collateral = outcome(log, base[start], tables)
        snaps = kyobo.build(books, tables['rcvd'], tables['rtgd'], tables['sales'], cuts)
        prec, missing = ei.score(snaps, truth)
        noise = ei.label_noise(snaps, clean_snaps['full'])
        for m, (p, ap) in prec.items():
            runs.append(dict(store=store, rate=rate, seed=seed, condition=cond, strategy=m, precision20=p, average_precision=ap,
                             missing_test=missing, collateral=collateral, label_flip=noise['flip'],
                             cost_usd=reviews.get(cond, {}).get('cost_usd', 0.0)))
        for s in states:
            det.append(dict(store=store, rate=rate, seed=seed, condition=cond, **s))
    return runs, det, reviews


def prepare(store):
    raw = ei.STORES[store]['load']()[:4]
    _, drop = kyobo.quality(*raw)
    clean = {n: kyobo.clean(*raw, drop, full) for n, full in (('minimal', False), ('full', True))}
    cuts = kyobo.cutoffs(clean['full'][1], clean['full'][3])
    snaps = {n: kyobo.build(*d, cuts) for n, d in clean.items()}
    truth = snaps['full'].query("split == 'test'")[['isbn', 'cutoff', 'label']].rename(columns={'label': 'true_label'})
    return raw, cuts, truth, snaps


def summarize(runs, det):
    runs, det = pd.DataFrame(runs), pd.DataFrame(det)
    perf = (runs.groupby(['store', 'rate', 'condition', 'strategy'])
            .agg(n=('seed', 'nunique'), precision20=('precision20', 'mean'), p_min=('precision20', 'min'), p_max=('precision20', 'max'),
                 average_precision=('average_precision', 'mean'), collateral=('collateral', 'mean'), label_flip=('label_flip', 'mean'), cost_usd=('cost_usd', 'mean'))
            .reset_index().to_dict('records'))
    d = det[det.state != 'missing'] if len(det) else det
    fixed = (d.assign(fixed=d.state.isin(['removed', 'restored']), restored=d.state.eq('restored'))
             .groupby(['condition', 'table', 'type']).agg(injected=('fixed', 'size'), fixed=('fixed', 'mean'), restored=('restored', 'mean'))
             .reset_index().to_dict('records')) if len(d) else []
    return dict(performance=perf, detection=fixed)


# ---------------------------------------------------------------- 보고서
pc = lambda v: '–' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v * 100:.1f}%'
TYPE_KO = {k: v[0] for k, v in ei.TYPES.items()}


def render(summary, reviews, config):
    t = common.table
    perf = [r for r in summary['performance'] if r['strategy'] == 'boosting']
    parts = ['<h1>LLM 에이전트 검토 — A/B/C/D 비교</h1>',
             f'<p class="note">실행 {html.escape(config["run_id"])} · 모델 {html.escape(config["model"])} (effort {config["effort"]}) · 오류 비율 {", ".join(pc(r) for r in config["rates"])} · 시드 {config["seeds"]}개 · '
             f'에이전트 비용 합계 ${config["cost_usd"]:.2f}{" (구독 사용: API 환산 추정치이며 실제 청구 아님)" if config["backend"] == "subscription" else ""}. 에이전트는 집계 도구 결과만 보고 원본 행은 보지 않는다.</p>',
             '<section><h2>조건</h2>' + t([dict(c=k, d=v) for k, v in CONDITIONS.items()], [('c', '조건', None), ('d', '내용', None)]) +
             '<p class="note">H는 조치 목록 일부를 정해진 값으로 일괄 적용한 대조군이다. 실험자가 오류 유형을 알고 고른 값이므로 상한에 가까운 참고치로만 본다.</p></section>']
    parts.append('<section><h2>1. 재주문 예측 Precision@20% (부스팅)</h2><p class="note">부수 피해 = 넣은 오류가 아닌 행을 지우거나 바꾼 수(시드 평균).</p>' +
                 t([dict(r, cond=f'{r["condition"]} {CONDITIONS[r["condition"]]}', range=f'[{pc(r["p_min"])}, {pc(r["p_max"])}]', col=f'{r["collateral"]:.0f}',
                         cost=f'${r["cost_usd"]:.2f}') for r in perf],
                   [('store', '서점', None), ('rate', '오류 비율', pc), ('cond', '조건', None), ('n', '시드', str), ('precision20', 'Precision@20%', pc),
                    ('range', '범위', str), ('average_precision', 'AP', pc), ('label_flip', '학습 라벨 뒤집힘', pc), ('col', '부수 피해(행)', str), ('cost', '비용/회', str)]) + '</section>')
    parts.append('<section><h2>2. 오류 유형별 처리율</h2><p class="note">처리 = 지움 또는 원래 값으로 되돌림. 되돌림은 지우지 않고 고친 비율. 행 누락은 행 단위로 처리할 수 없어 뺐다.</p>' +
                 t([dict(r, cond=f'{r["condition"]} {CONDITIONS[r["condition"]]}', tbl=ag.TABLE_KO[r['table']], tp=TYPE_KO[r['type']]) for r in summary['detection']],
                   [('cond', '조건', None), ('tbl', '표', None), ('tp', '오류', None), ('injected', '넣은 행', str), ('fixed', '처리', pc), ('restored', '되돌림', pc)]) + '</section>')
    rows = []
    for key, rv in reviews.items():
        for a, ap in zip(rv['actions'], rv['applied']):
            rows.append(dict(run=key, action=a['action'], table=ag.TABLE_KO[a['table']], params=json.dumps({k: a[k] for k in ('keys', 'threshold', 'window_months', 'max_days', 'require_multiple_of_10') if a[k]}, ensure_ascii=False),
                             changed=f'지움 {ap["removed"]} · 바꿈 {ap["modified"]}', why=a['rationale'][:160]))
    parts.append('<section><h2>3. 에이전트가 최종 승인한 조치</h2>' + t(rows, [('run', '실행', None), ('action', '조치', None), ('table', '표', None), ('params', '매개변수', None),
                                                                      ('changed', '적용 결과', None), ('why', '근거 요약', None)]) + '</section>')
    parts.append('<section><h2>한계</h2><ul><li>에이전트가 고를 수 있는 조치 목록은 실험자가 만들었다. 목록에 없는 방식의 수정은 할 수 없다.</li>'
                 '<li>시드가 적어 성능 차이의 불확실성이 크다. 처리율·부수 피해가 더 안정적인 지표다.</li>'
                 '<li>교보·예스24 두 서점 자료만 쓴다. 출판사 거래처 장부는 포함하지 않는다.</li></ul></section>')
    return f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>에이전트 검토 실험</title><style>{common.CSS}</style><body>{"".join(parts)}</body></html>'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--store', choices=[*ei.STORES, 'all'], default='all')
    p.add_argument('--rates', default='0,0.1,0.2')
    p.add_argument('--seeds', type=int, default=2)
    p.add_argument('--model', default='claude-opus-5-5')
    p.add_argument('--effort', default='medium', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    p.add_argument('--backend', choices=['subscription', 'api'], default='subscription', help='subscription: Claude Code 로그인(구독), api: ANTHROPIC_API_KEY')
    p.add_argument('--max-cost', type=float, default=30.0, help='누적 비용(구독이면 API 환산 추정치, USD)이 넘으면 남은 실행을 멈춘다')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--no-llm', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT.parent / '결과')
    args = p.parse_args()
    if args.pilot:
        args.store, args.rates, args.seeds = '교보', '0.2', 1
    rates = [float(r) for r in args.rates.split(',')]
    args.output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.output / 'agent.sqlite')
    if args.no_llm:
        agents = None
    elif args.backend == 'api':
        agents = ag.ApiAgents(ag.make_client(), args.model, args.effort, db)
    else:
        agents = ag.SubscriptionAgents(args.model, args.effort, db)
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    out = args.output / run_id
    out.mkdir()
    logf = open(out / 'run.log', 'a', encoding='utf8')

    def log(msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n'); logf.flush()

    config = dict(run_id=run_id, model=args.model, effort=args.effort, backend='none' if args.no_llm else args.backend, rates=rates, seeds=args.seeds,
                  heuristic=HEURISTIC, cost_usd=0.0, stopped=None, seconds=0)
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding='utf8')
    log(f'시작: {run_id} · 백엔드 {config["backend"]} · 모델 {args.model}')
    started, runs, det, reviews = time.time(), [], [], {}

    def save():
        summary = summarize(runs, det)
        config.update(cost_usd=agents.cost if agents else 0.0, seconds=round(time.time() - started))
        pd.DataFrame(runs).to_csv(out / 'runs.csv', index=False, encoding='utf-8-sig')
        (out / 'results.json').write_text(json.dumps(dict(config=config, **summary, reviews=reviews), ensure_ascii=False, indent=1, default=str), encoding='utf8')
        (out / 'index.html').write_text(render(summary, reviews, config), encoding='utf8')

    for store in (ei.STORES if args.store == 'all' else [args.store]):
        raw, cuts, truth, snaps = prepare(store)
        for rate in rates:
            for seed in (range(1) if rate == 0 else range(args.seeds)):
                if agents is not None and agents.cost > args.max_cost:
                    config['stopped'] = f'누적 비용 ${agents.cost:.2f} > ${args.max_cost}'
                    break
                r, d, rv = one(store, raw, cuts, truth, snaps, rate, seed, agents, run_id, out, log)
                runs += r; det += d
                reviews.update({f'{store}/{rate:.0%}/s{seed}/{k}': v for k, v in rv.items()})
                save()
                log(f'{store} {rate:.0%} s{seed} 완료 · 누적 비용 ${agents.cost if agents else 0:.2f}')
    save()
    db.close()
    if (args.output / 'agent.sqlite').exists():
        shutil.copy2(args.output / 'agent.sqlite', out / 'agent.sqlite')
    log(f'끝: {config["seconds"]}s · 비용 ${config["cost_usd"]:.2f}' + (f' · 중단: {config["stopped"]}' if config['stopped'] else ''))
    logf.close()
    files = sorted(f for f in out.rglob('*') if f.is_file() and f.name != 'manifest.json')
    (out / 'manifest.json').write_text(json.dumps(dict(run_id=run_id, raw_data_included=False,
        files=[dict(path=f.relative_to(out).as_posix(), bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]),
        ensure_ascii=False, indent=1), encoding='utf8')
    print(out / 'index.html')


if __name__ == '__main__':
    main()
