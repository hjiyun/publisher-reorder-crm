"""시나리오 실험. 튜닝(dev)과 최종 평가(final)의 시나리오·시드를 나눈다.

실행:  python experiment.py dev       제어 강도 격자, 보고 지연별 실패 분석 → ../결과/<run_id>/
       python experiment.py final     dev에서 고정한 설정으로 별도 시나리오·시드 평가
"""
import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import sim
from sim import Asap, Config, EvenHourly, Feedback, Scenario, make_world, metrics, simulate, summarize

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / '결과'

DEV_SEEDS = range(40)
FINAL_SEEDS = range(1000, 1200)
FINAL_SMALL_SEEDS = range(1000, 3000)  # 소액 예산은 하루 전환이 1건도 안 돼 시드를 늘린다
KP_GRID = [0.5, 1, 2, 4, 8, 16, 32]
KI_GRID = [0, 0.02, 0.1, 0.5]
LAG_GRID = [0, 1, 4, 8]  # 0분, 15분, 1시간, 2시간
LARGE = Config(budget=1_000_000)
SMALL = Config(budget=100_000 / 30)  # 교보 ROI 2의 시리즈 월 10만원 광고를 하루로 나눈 값
MARGIN = 4600  # 교보 분석의 권당 공헌이익 중앙값. 본전 CPA 기준선으로만 쓴다

DEV = [Scenario('normal', '평상시'),
       Scenario('surge', '트래픽 급증', traffic=((10, 13, 2.0),)),
       Scenario('evening_cvr', '저녁 전환율 상승', cvr=((18, 23, 1.5),)),
       Scenario('drop', '유입 급감', traffic=((14, 18, 0.4),))]

FINAL = [Scenario('normal', '평상시'),
         Scenario('flat', '대조군: 시간대 차이 없음', flat=True),
         Scenario('surge_late', '저녁 트래픽 급증', traffic=((19, 22, 2.5),)),
         Scenario('evening_cvr_price', '저녁 전환율·가격 상승', cvr=((18, 24, 2.0),), price=((18, 24, 1.2),)),
         Scenario('drop_morning', '오전 유입 급감', traffic=((8, 12, 0.2),)),
         Scenario('surge_then_drop', '급증 후 급감', traffic=((9, 12, 2.0), (15, 20, 0.3))),
         Scenario('low_day', '종일 유입 절반', traffic=((0, 24, 0.5),)),
         Scenario('morning_cvr', '오전 전환 우세(가정 반전)', cvr=((6, 12, 1.6), (18, 24, 0.8))),
         Scenario('surge_lag', '급증 + 보고 2시간 지연', traffic=((11, 14, 2.0),), lag=8),
         Scenario('drop_lag', '급감 + 보고 2시간 지연', traffic=((9, 13, 0.3),), lag=8)]


def evaluate(make_policies, scenarios, seeds, cfg, lag=None):
    """정책·시나리오·시드별 지표 표. 같은 시드의 세계를 모든 정책이 공유한다."""
    rows = []
    for sc in scenarios:
        for seed in seeds:
            world = make_world(cfg, sc, seed)
            for name, policy in make_policies().items():
                r = simulate(policy, world, cfg.budget, sc.lag if lag is None else lag)
                rows.append(dict(scenario=sc.name, label=sc.label, seed=seed, policy=name, **metrics(r, sc.onset)))
    return pd.DataFrame(rows)


def new_run():
    run_id = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    out = RESULTS / run_id
    out.mkdir(parents=True)
    return run_id, out


def tune(compensate='none'):
    """지연 15분 기본 조건에서 제어 강도를 고른다.

    기준은 실행 전에 정했다: 네 시나리오 평균 목표 오차가 최소인 조합. 오차가 0.1%p 안이면 변동성이 낮은 쪽.
    """
    rows = []
    for kp in KP_GRID:
        for ki in KI_GRID:
            df = evaluate(lambda: {'fb': Feedback(kp, ki, compensate=compensate)}, DEV, DEV_SEEDS, LARGE)
            rows.append(dict(kp=kp, ki=ki, tracking=df.tracking.mean(), volatility=df.volatility.mean(), conv=df.conv.mean(), burn=df.burn.mean()))
    grid = pd.DataFrame(rows)
    close = grid[grid.tracking <= grid.tracking.min() + 0.001]
    best = close.sort_values('volatility').iloc[0]
    return grid, float(best.kp), float(best.ki)


def run_dev():
    run_id, out = new_run()
    grid, kp, ki = tune()
    grid.to_csv(out / 'dev_grid.csv', index=False)
    print(grid.round(4).to_string())
    print('선택', kp, ki)
    grid2, kp2, ki2 = tune('predict')
    grid2.to_csv(out / 'dev_grid_predict.csv', index=False)
    print('지연 보정판 선택', kp2, ki2)
    # 소액 예산에서 제어 강도별 변화. 낙찰 1건이 예산의 몇 %를 차지해 제어가 흔들린다
    rows = []
    for k in KP_GRID:
        df = evaluate(lambda: {'fb': Feedback(k, ki2, compensate='predict')}, DEV, range(400), SMALL)
        rows.append(dict(kp=k, tracking=df.tracking.mean(), volatility=df.volatility.mean(), saturated=df.saturated.mean(), burn=df.burn.mean(), conv=df.conv.mean()))
    small_gain = pd.DataFrame(rows)
    small_gain.to_csv(out / 'dev_small_gain.csv', index=False)
    print(small_gain.round(3).to_string())
    # 제어 강도 × 보고 지연 × 지연 보정 방식. 고른 값이 지연이 길어질 때 어떻게 무너지는지 본다
    rows = []
    for compensate in ('none', 'shift', 'predict'):
        for lag in LAG_GRID:
            for k in KP_GRID:
                df = evaluate(lambda: {'fb': Feedback(k, ki, compensate=compensate)}, DEV, DEV_SEEDS, LARGE, lag=lag)
                rows.append(dict(compensate=compensate, lag=lag, kp=k, tracking=df.tracking.mean(), peak_err=df.peak_err.mean(), volatility=df.volatility.mean(),
                                 saturated=df.saturated.mean(), burn=df.burn.mean(), conv=df.conv.mean()))
    lagmap = pd.DataFrame(rows)
    lagmap.to_csv(out / 'dev_lag.csv', index=False)
    print(lagmap.round(3).to_string())
    base = lambda: {'asap': Asap(), 'even': EvenHourly(), 'fb': Feedback(kp, ki)}
    cols = ['burn', 'tracking', 'peak_err', 'conv', 'cpa', 'exhausted', 'exhaust_h', 'volatility', 'saturated', 'recovery_h']
    for name, cfg in (('large', LARGE), ('small', SMALL)):
        df = evaluate(base, DEV, DEV_SEEDS, cfg)
        df.to_csv(out / f'dev_{name}.csv', index=False)
        print(name)
        print(summarize(df)[cols].round(3).to_string())
    (out / 'manifest.json').write_text(json.dumps(dict(run_id=run_id, stage='dev', kp=kp, ki=ki, kp_predict=kp2, ki_predict=ki2, seeds=len(DEV_SEEDS), kp_grid=KP_GRID, ki_grid=KI_GRID, lag_grid=LAG_GRID), ensure_ascii=False, indent=1), encoding='utf8')
    print(out)


def latest_dev():
    """가장 최근 dev 실행의 고정 설정. final은 이 값만 쓰고 다시 튜닝하지 않는다."""
    runs = sorted(p for p in RESULTS.glob('*/manifest.json') if json.loads(p.read_text(encoding='utf8'))['stage'] == 'dev')
    return json.loads(runs[-1].read_text(encoding='utf8'))


def frozen_policies(dev):
    return {'asap': Asap(), 'even': EvenHourly(), 'fb': Feedback(dev['kp'], dev['ki']),
            'fb_v2': Feedback(dev['kp_predict'], dev['ki_predict'], compensate='predict')}


def paired_vs_asap(df):
    """같은 시드끼리 묶은 전환 수 차이(정책 − 소진까지 집행)와 95% 구간."""
    wide = df.pivot_table(index=['scenario', 'seed'], columns='policy', values='conv')
    rows = []
    for sc, g in wide.groupby('scenario', sort=False):
        for name in ('even', 'fb', 'fb_v2'):
            d = g[name] - g['asap']
            se = d.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(scenario=sc, policy=name, diff=d.mean(), low=d.mean() - 1.96 * se, high=d.mean() + 1.96 * se, pct=d.mean() / g['asap'].mean()))
    return pd.DataFrame(rows)


def days_to_detect(conv, lift=0.1):
    """실제로 전환이 lift만큼 다를 때, 정책을 날짜 단위로 나눠 집행해 검출(양측 5%, 검정력 80%)하는 데 필요한 정책당 일수."""
    return float(2 * (1.96 + 0.84) ** 2 * conv.var(ddof=1) / (lift * conv.mean()) ** 2)


def run_final():
    dev = latest_dev()
    run_id, out = new_run()
    report = dict(run_id=run_id, stage='final', dev_run=dev['run_id'], seeds=dict(large=len(FINAL_SEEDS), small=len(FINAL_SMALL_SEEDS)), settings={k: dev[k] for k in ('kp', 'ki', 'kp_predict', 'ki_predict')})
    cols = ['burn', 'tracking', 'peak_err', 'conv', 'cpa', 'exhausted', 'exhaust_h', 'volatility', 'saturated', 'recovery_h', 'never_recovered']
    for name, cfg, seeds in (('large', LARGE, FINAL_SEEDS), ('small', SMALL, FINAL_SMALL_SEEDS)):
        df = evaluate(lambda: frozen_policies(dev), FINAL, seeds, cfg)
        df.to_csv(out / f'final_{name}.csv', index=False)
        table = summarize(df)
        table.to_csv(out / f'final_summary_{name}.csv')
        diffs = paired_vs_asap(df)
        diffs.to_csv(out / f'final_diff_{name}.csv', index=False)
        print(name)
        print(table[cols].round(3).to_string())
        print(diffs.round(3).to_string())
        normal = df[df.scenario == 'normal']
        asap = normal[normal.policy == 'asap'].conv
        report[name] = dict(budget=cfg.budget, conv_per_day=float(asap.mean()), days_for_10pct=days_to_detect(asap))
    mean_price = LARGE.cpc * np.exp(LARGE.price_sigma ** 2 / 2)
    report['breakeven'] = dict(margin=MARGIN, mean_price=mean_price, cvr_assumed=LARGE.cvr, cpa_assumed=mean_price / LARGE.cvr, cvr_needed=mean_price / MARGIN)
    (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['dev', 'final'])
    args = ap.parse_args()
    run_dev() if args.stage == 'dev' else run_final()
