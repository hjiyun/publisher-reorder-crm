"""광고 예산 페이싱 가상 환경과 정책. 실제 광고 데이터를 쓰지 않는다.

하루를 15분 구간 96개로 나눈다. 구간마다 광고 기회가 도착하고, 정책은 참여 확률 a(0~1)를 정한다.
참여한 기회는 낙찰로 보고 가격을 지불하며, 시간대별 전환 확률로 전환된다. 예산을 넘는 집행은 막는다.
같은 시드는 같은 기회·가격·난수를 쓰므로 정책 차이에 도착 난수 차이가 섞이지 않는다.

모든 수치(유입 곡선, 가격, 전환율)는 가정값이다. 결과는 가상 환경에서의 정책 비교로만 해석한다.
"""
from dataclasses import dataclass

import numpy as np

STEPS_PER_HOUR = 4
T = 24 * STEPS_PER_HOUR
HOURS = np.arange(T) / STEPS_PER_HOUR

# 시간대별 기준 곡선(가정). 정책은 이 곡선만 예측값으로 안다.
TRAFFIC_BY_HOUR = np.array([.25, .15, .1, .1, .1, .2, .4, .7, 1., 1.1, 1.1, 1.2, 1.4, 1.2, 1.1, 1.1, 1.1, 1.2, 1.3, 1.5, 1.7, 1.7, 1.2, .6])
CVR_BY_HOUR = np.array([1.] * 18 + [1.1, 1.2, 1.2, 1.2, 1.1, 1.])
PRICE_BY_HOUR = np.array([.9] * 7 + [1.] * 12 + [1.1, 1.1, 1.1, 1.] + [.9])


@dataclass(frozen=True)
class Config:
    budget: float = 1_000_000      # 하루 예산(원)
    cpc: float = 200.0             # 기회 1건 가격 중앙값(원)
    price_sigma: float = 0.5       # 가격 로그정규 표준편차
    cvr: float = 0.02              # 기준 전환 확률
    supply_ratio: float = 2.5      # 전 기회에 참여할 때의 예상 지출 ÷ 예산
    daily_sigma: float = 0.1       # 하루 유입 수준의 예측 오차(로그정규)


@dataclass(frozen=True)
class Scenario:
    """기준 곡선에 곱할 시간대 구간과 보고 지연. 구간은 (시작시, 끝시, 배수)."""
    name: str
    label: str
    traffic: tuple = ()
    cvr: tuple = ()
    price: tuple = ()
    lag: int = 1                   # 지출 보고 지연(구간 수). 0 = 즉시, 1 = 15분, 8 = 2시간
    flat: bool = False             # True면 시간대별 전환율·가격 차이를 없앤 대조군
    cvr_curve: tuple = ()          # 지정하면 CVR_BY_HOUR 대신 쓴다


    @property
    def onset(self):
        """가장 이른 교란 시각. 교란이 없으면 None이다."""
        starts = [w[0] for w in self.traffic + self.cvr + self.price]
        return min(starts) if starts else None


def _windows(windows):
    m = np.ones(T)
    for start, end, k in windows:
        m[(HOURS >= start) & (HOURS < end)] *= k
    return m


def _expand(by_hour):
    return np.repeat(by_hour, STEPS_PER_HOUR)


def forecast_shape():
    """정책이 아는 예측 지출 곡선. 구간별 지출 비중으로 합은 1이다. 목표 지출 곡선의 근거가 된다."""
    s = _expand(TRAFFIC_BY_HOUR) * _expand(PRICE_BY_HOUR)
    return s / s.sum()


def target_curve(budget):
    """구간 t가 끝났을 때의 목표 누적 지출."""
    return budget * np.cumsum(forecast_shape())


def make_world(cfg, sc, seed):
    """시드 하나의 하루 세계. 구간별 가격·참여 난수·전환 여부를 미리 뽑아 둔다."""
    rng = np.random.default_rng(seed)
    traffic = _expand(TRAFFIC_BY_HOUR) * _windows(sc.traffic)
    price_m = _windows(sc.price) * (np.ones(T) if sc.flat else _expand(PRICE_BY_HOUR))
    base_cvr = np.ones(T) if sc.flat else _expand(np.array(sc.cvr_curve) if sc.cvr_curve else CVR_BY_HOUR)
    cvr = np.clip(cfg.cvr * base_cvr * _windows(sc.cvr), 0, 1)
    # 기준 곡선에서 전 기회 참여 지출이 supply_ratio × 예산이 되도록 기회 수를 맞춘다
    mean_price = cfg.cpc * np.exp(cfg.price_sigma ** 2 / 2)
    base_spend = (_expand(TRAFFIC_BY_HOUR) * _expand(PRICE_BY_HOUR)).sum() * mean_price
    scale = cfg.supply_ratio * cfg.budget / base_spend
    daily = rng.lognormal(-cfg.daily_sigma ** 2 / 2, cfg.daily_sigma)
    counts = rng.poisson(traffic * scale * daily)
    steps = []
    for t, n in enumerate(counts):
        prices = cfg.cpc * price_m[t] * rng.lognormal(0, cfg.price_sigma, n)
        steps.append((prices, rng.random(n), rng.random(n) < cvr[t]))
    return steps


# ---------------------------------------------------------------- 정책
class Policy:
    """act(t, observed)는 참여 확률을 돌려준다. observed는 보고 지연이 적용된 누적 지출이다.

    hourly_cap은 시간당 지출 상한이다. 하루 예산과 마찬가지로 집행 시스템이 실제 지출로 막는다.
    """
    label = ''
    hourly_cap = None

    def reset(self, budget, lag):
        self.budget, self.lag = budget, lag

    def act(self, t, observed):
        raise NotImplementedError


class Asap(Policy):
    """예산이 떨어질 때까지 모든 기회에 참여한다."""
    label = '소진까지 집행'

    def act(self, t, observed):
        return 1.0


class EvenHourly(Policy):
    """하루 예산을 24시간에 똑같이 나눈다. 쓰지 못한 시간 예산은 넘기지 않는다."""
    label = '시간대별 동일 할당'

    def reset(self, budget, lag):
        super().reset(budget, lag)
        self.hourly_cap = budget / 24

    def act(self, t, observed):
        return 1.0


class Feedback(Policy):
    """목표 누적 지출과 실제 지출의 차이로 참여 확률을 조정한다(PI 제어).

    a = a0 + Kp·e + Ki·Σe, e = (목표 − 지출 추정) ÷ 예산, a0 = 1 ÷ supply_ratio(예측 기준 정상 참여율).
    compensate는 보고 지연을 다루는 방식이다.
      none     지금 시점 목표와 지연된 지출을 그대로 비교한다
      shift    목표도 지연만큼 과거 값으로 맞춰 비교한다
      predict  아직 보고되지 않은 구간의 지출을 자기 참여 확률과 예측 곡선으로 추정해 더한다
    """
    label = '목표 곡선 추종'

    def __init__(self, kp=8.0, ki=0.0, a0=0.4, compensate='none'):
        self.kp, self.ki, self.a0, self.compensate = kp, ki, a0, compensate

    def reset(self, budget, lag):
        super().reset(budget, lag)
        self.target = target_curve(budget)
        self.full = budget * forecast_shape() / self.a0  # 전 기회 참여 시 예상 지출
        self.integral = 0.0
        self.history = np.zeros(T)

    def _gap(self, t, observed):
        now = self.target[t - 1] if t > 0 else 0.0
        if self.compensate == 'shift':
            return (self.target[t - 1 - self.lag] if t - 1 - self.lag >= 0 else 0.0) - observed
        if self.compensate == 'predict':
            first = max(t - self.lag, 0)
            return now - observed - (self.history[first:t] * self.full[first:t]).sum()
        return now - observed

    def act(self, t, observed):
        e = self._gap(t, observed) / self.budget
        self.integral += e
        a = float(np.clip(self.a0 + self.kp * e + self.ki * self.integral, 0, 1))
        self.history[t] = a
        return a


# ---------------------------------------------------------------- 실행
@dataclass
class Run:
    spend: np.ndarray              # 구간 t까지의 누적 지출
    conv: np.ndarray               # 구간 t까지의 누적 전환
    a: np.ndarray                  # 구간별 참여 확률
    budget: float


def _buy(prices, converts, cap):
    """상한 안에서 도착 순서대로 산다. 남은 돈보다 비싼 기회는 건너뛰고 다음 기회를 본다."""
    paid = np.cumsum(prices)
    if len(paid) == 0 or paid[-1] <= cap:
        return (float(paid[-1]) if len(paid) else 0.0), int(converts.sum())
    spent, got = 0.0, 0
    for price, c in zip(prices, converts):
        if spent + price <= cap:
            spent += price
            got += int(c)
    return spent, got


def simulate(policy, world, budget, lag=1):
    policy.reset(budget, lag)
    spend = np.zeros(T)
    conv = np.zeros(T)
    actions = np.zeros(T)
    total, total_conv, hour_spent = 0.0, 0, 0.0
    for t, (prices, u, converts) in enumerate(world):
        if t % STEPS_PER_HOUR == 0:
            hour_spent = 0.0
        observed = spend[t - 1 - lag] if t - 1 - lag >= 0 else 0.0
        a = policy.act(t, observed)
        actions[t] = a
        cap = budget - total
        if policy.hourly_cap is not None:
            cap = min(cap, policy.hourly_cap - hour_spent)
        if len(prices) and a > 0 and cap > 0:
            joined = u < a
            spent, got = _buy(prices[joined], converts[joined], cap)
            total += spent
            total_conv += got
            hour_spent += spent
        spend[t], conv[t] = total, total_conv
    return Run(spend, conv, actions, budget)


def metrics(run, recovery_from=None, tol=0.03):
    """정책 비교 지표. recovery_from(시)을 주면 그 시각 이후 목표 오차가 tol 안으로 돌아오기까지의 시간을 잰다."""
    target = target_curve(run.budget)
    err = (run.spend - target) / run.budget
    burn = run.spend[-1] / run.budget
    conv = run.conv[-1]
    exhausted = np.flatnonzero(run.spend >= run.budget * 0.95)  # 소액 예산은 마지막 몇 원을 쓸 수 없어 95%로 잰다
    da = np.abs(np.diff(run.a))
    out = dict(spend=float(run.spend[-1]), burn=burn, tracking=float(np.abs(err).mean()), peak_err=float(np.abs(err).max()),
               conv=float(conv),
               exhaust_h=float(HOURS[exhausted[0]] + 1 / STEPS_PER_HOUR) if len(exhausted) else np.nan,
               volatility=float(da.mean()), saturated=float(((run.a <= 0) | (run.a >= 1)).mean()))
    if recovery_from is not None:
        # 교란 시작부터, 오차가 tol 안으로 들어와 하루 끝까지 머물게 되기까지의 시간. 끝내 못 돌아오면 nan이다
        late = np.abs(err[int(recovery_from * STEPS_PER_HOUR):]) <= tol
        bad = np.flatnonzero(~late)
        out['recovery_h'] = np.nan if len(bad) and bad[-1] == len(late) - 1 else float(bad[-1] + 1 if len(bad) else 0) / STEPS_PER_HOUR
    return out


def summarize(df, by=('scenario', 'policy')):
    """시드 평균 표. CPA는 일별 CPA의 평균이 아니라 총지출 ÷ 총전환으로 낸다(전환 0건인 날이 빠지지 않게)."""
    g = df.groupby(list(by), sort=False)
    out = g[['burn', 'tracking', 'peak_err', 'conv', 'volatility', 'saturated']].mean()
    out['cpa'] = g.spend.sum() / g.conv.sum()
    out['exhausted'] = g.exhaust_h.apply(lambda v: v.notna().mean())
    out['exhaust_h'] = g.exhaust_h.mean()
    if 'recovery_h' in df:
        out['recovery_h'] = g.recovery_h.mean()
        out['never_recovered'] = g.recovery_h.apply(lambda v: v.isna().mean())
    return out
