import unittest

import numpy as np
import pandas as pd

import sim
from sim import Asap, Config, EvenHourly, Feedback, Policy, Run, Scenario, make_world, metrics, simulate, summarize, target_curve

CFG = Config(budget=100_000)
BASE = Scenario('normal', '평상시')


class Spy(Policy):
    """관측값을 기록하고 정해진 참여 확률로 집행한다."""

    def __init__(self, a=0.5):
        self.a = a
        self.seen = []

    def act(self, t, observed):
        self.seen.append(observed)
        return self.a


class EnvTest(unittest.TestCase):
    def test_target_curve_rises_to_budget(self):
        t = target_curve(1000)
        self.assertTrue((np.diff(t) > 0).all())
        self.assertAlmostEqual(t[-1], 1000)

    def test_same_seed_same_world(self):
        a, b = make_world(CFG, BASE, 7), make_world(CFG, BASE, 7)
        for (p1, u1, c1), (p2, u2, c2) in zip(a, b):
            np.testing.assert_array_equal(p1, p2)
            np.testing.assert_array_equal(u1, u2)
            np.testing.assert_array_equal(c1, c2)

    def test_supply_ratio_sets_full_spend(self):
        # 전 기회 참여 지출의 평균이 예산 × supply_ratio에 가까워야 한다
        full = np.mean([sum(p.sum() for p, _, _ in make_world(CFG, BASE, s)) for s in range(200)])
        self.assertAlmostEqual(full / CFG.budget, CFG.supply_ratio, delta=0.1)

    def test_surge_and_drop_change_opportunities(self):
        count = lambda sc: sum(len(p) for p, _, _ in make_world(CFG, sc, 1)[40:52])
        base = count(BASE)
        self.assertGreater(count(Scenario('s', 's', traffic=((10, 13, 2.0),))), base * 1.5)
        self.assertLess(count(Scenario('d', 'd', traffic=((10, 13, 0.4),))), base * 0.6)

    def test_higher_cvr_never_loses_conversions(self):
        # 같은 시드에서 지출 경로가 같으므로 전환율이 오르면 전환이 줄 수 없다
        up = Scenario('u', 'u', cvr=((18, 23, 1.5),))
        for seed in range(30):
            r0 = simulate(Feedback(), make_world(CFG, BASE, seed), CFG.budget)
            r1 = simulate(Feedback(), make_world(CFG, up, seed), CFG.budget)
            np.testing.assert_array_equal(r0.spend, r1.spend)
            self.assertGreaterEqual(r1.conv[-1], r0.conv[-1])


class ExecutionTest(unittest.TestCase):
    def test_budget_never_exceeded(self):
        for cfg in (CFG, Config(budget=3333)):
            for sc in (BASE, Scenario('s', 's', traffic=((10, 13, 3.0),))):
                for policy in (Asap(), EvenHourly(), Feedback(), Feedback(compensate='predict')):
                    r = simulate(policy, make_world(cfg, sc, 3), cfg.budget)
                    self.assertLessEqual(r.spend[-1], cfg.budget + 1e-6)
                    self.assertTrue((np.diff(r.spend) >= 0).all())

    def test_even_hourly_respects_hour_cap(self):
        r = simulate(EvenHourly(), make_world(CFG, BASE, 3), CFG.budget, lag=8)
        by_hour = np.diff(np.r_[0, r.spend[3::4]])
        self.assertLessEqual(by_hour.max(), CFG.budget / 24 + 1e-6)

    def test_policy_sees_only_delayed_spend(self):
        for lag in (0, 1, 8):
            spy = Spy()
            r = simulate(spy, make_world(CFG, BASE, 3), CFG.budget, lag=lag)
            for t, seen in enumerate(spy.seen):
                expected = r.spend[t - 1 - lag] if t - 1 - lag >= 0 else 0.0
                self.assertEqual(seen, expected)

    def test_buy_skips_unaffordable_and_continues(self):
        spent, got = sim._buy(np.array([50.0, 500.0, 40.0]), np.array([True, True, True]), 100)
        self.assertEqual((spent, got), (90.0, 2))

    def test_compensation_is_identical_without_lag(self):
        world = make_world(CFG, BASE, 3)
        spends = [simulate(Feedback(compensate=c), world, CFG.budget, lag=0).spend for c in ('none', 'shift', 'predict')]
        np.testing.assert_array_almost_equal(spends[0], spends[1])
        np.testing.assert_array_almost_equal(spends[0], spends[2])

    def test_asap_exhausts_before_paced_policy(self):
        world = make_world(CFG, BASE, 3)
        fast = metrics(simulate(Asap(), world, CFG.budget))
        paced = metrics(simulate(Feedback(), world, CFG.budget))
        self.assertLess(fast['exhaust_h'], 18)
        self.assertGreater(paced['exhaust_h'], 21)  # 목표 곡선이 95%에 닿는 시각이 22시 부근이다


class ControlTest(unittest.TestCase):
    def test_flat_world_gives_no_timing_advantage(self):
        # 시간대 차이가 없으면 예산을 다 쓰는 정책끼리 전환 수가 같아야 한다. 차이가 나면 환경이나 집행 버그다
        flat = Scenario('flat', '대조군', flat=True)
        diffs = []
        for seed in range(150):
            world = make_world(CFG, flat, seed)
            diffs.append(simulate(Feedback(), world, CFG.budget).conv[-1] - simulate(Asap(), world, CFG.budget).conv[-1])
        se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
        self.assertLess(abs(np.mean(diffs)), 2.6 * se)


class MetricTest(unittest.TestCase):
    def run_with(self, spend):
        return Run(np.asarray(spend, float), np.linspace(0, 10, sim.T), np.full(sim.T, 0.5), 1000.0)

    def test_on_target_has_no_error(self):
        m = metrics(self.run_with(target_curve(1000)), recovery_from=10)
        self.assertAlmostEqual(m['tracking'], 0)
        self.assertEqual(m['recovery_h'], 0)

    def test_recovery_time_after_disturbance(self):
        spend = target_curve(1000).copy()
        spend[40:48] -= 100  # 10시부터 2시간 동안 10% 뒤처짐
        self.assertEqual(metrics(self.run_with(spend), recovery_from=10)['recovery_h'], 2)

    def test_recovery_is_nan_when_never_back(self):
        spend = target_curve(1000).copy()
        spend[40:] -= 100
        self.assertTrue(np.isnan(metrics(self.run_with(spend), recovery_from=10)['recovery_h']))

    def test_cpa_uses_totals_not_daily_average(self):
        # 전환 0건인 날의 지출도 CPA에 들어가야 한다
        df = pd.DataFrame(dict(scenario='a', policy='p', spend=[100.0, 100.0], conv=[1.0, 0.0], burn=1.0, tracking=0.0, peak_err=0.0, volatility=0.0, saturated=0.0, exhaust_h=[1.0, np.nan]))
        row = summarize(df).iloc[0]
        self.assertEqual(row.cpa, 200.0)
        self.assertEqual(row.exhausted, 0.5)


if __name__ == '__main__':
    unittest.main()
