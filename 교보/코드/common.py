"""교보 분석 공통 부품. customer_retail의 pipeline.py·publisher.py·report.py에서 교보 분석에 필요한 것만 그대로 옮겨 왔다."""


import html
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# 분석 대상 서점. yes24.py가 바꿔 쓴다.
STORE = dict(name='교보', source='교보문고 협력사네트워크 SCM (2026-09-28 수집)', event='입하', reader='교보 구매자 집계')
CONTACT_FRACTION = 0.2
GAIN_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
FREQ_BINS = [1, 2, 3, 5, 10]  # F점수 1..5의 하한



# ---------------------------------------------------------------- 지표·모델·세분화 (pipeline.py)


def metrics(y, scores, fraction=CONTACT_FRACTION):
    y, scores = np.asarray(y), np.asarray(scores, float)
    if not len(y) or len(y) != len(scores) or not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid predictions')
    # 동점은 입력 순서(고객 ID 오름차순)로 고정한다. 모든 전략에 같은 규칙.
    rank = np.argsort(-scores, kind='stable')
    k = max(1, math.ceil(len(y) * fraction))
    positives, base = float(y.sum()), float(y.mean())
    precision = float(y[rank[:k]].mean())
    ends = np.r_[np.where(np.diff(scores[rank]) != 0)[0], len(y) - 1]
    tp = np.cumsum(y[rank])[ends]
    fp = (ends + 1) - tp
    ap = float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / positives) if positives else None
    auc = None
    if 0 < positives < len(y):
        trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # numpy 1.x / 2.x 호환
        auc = float(trapezoid(np.r_[0, tp] / positives, np.r_[0, fp] / (len(y) - positives)))
    return dict(n=len(y), selected=k, base_rate=base, precision20=precision,
                recall20=float(y[rank[:k]].sum() / positives) if positives else None,
                lift20=precision / base if base else None, average_precision=ap, roc_auc=auc)


def wilson(successes, n, z=1.96):
    if n == 0:
        return None, None
    p = successes / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def bootstrap_precision(y, scores, reps=1000, seed=0):
    rng, y, scores = np.random.default_rng(seed), np.asarray(y), np.asarray(scores, float)
    k, out = max(1, math.ceil(len(y) * CONTACT_FRACTION)), []
    for _ in range(reps):
        idx = np.sort(rng.integers(0, len(y), len(y)))
        top = np.argsort(-scores[idx], kind='stable')[:k]
        out.append(y[idx][top].mean())
    return float(np.quantile(out, .025)), float(np.quantile(out, .975))


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_models():
    # 설정은 사전에 고정한다. 평가 데이터로 조정하지 않는다.
    return {
        'logistic': make_pipeline(FunctionTransformer(signed_log), SimpleImputer(strategy='median', add_indicator=True),
                                  StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        'boosting': HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
    }


class RFM:
    """R·M은 학습 구간 5분위, F는 고정 구간. 점수 합이 같으면 최근 구매 고객이 먼저다."""

    def fit(self, train):
        q = [.2, .4, .6, .8]
        self.r_edges, self.m_edges = np.quantile(train.recency_days, q), np.quantile(train.monetary, q)
        return self

    def scores(self, df):
        r = 5 - np.searchsorted(self.r_edges, df.recency_days, side='left')
        f = np.searchsorted(FREQ_BINS, df.frequency, side='right').clip(1, 5)
        m = 1 + np.searchsorted(self.m_edges, df.monetary, side='right')
        return pd.DataFrame(dict(R=r, F=f, M=m), index=df.index)

    def predict(self, df):
        s = self.scores(df)
        return (s.R + s.F + s.M - df.recency_days / 1e5).to_numpy(float)

    def segment(self, df):
        s = self.scores(df)
        return pd.Series(np.select(
            [(s.R >= 4) & (s.F >= 4), (s.R >= 4), (s.R == 3), (s.F >= 4)],
            ['핵심 (최근·고빈도)', '최근·저빈도', '둔화 (중간 최근성)', '이탈 위험 우수고객'], '휴면'), index=df.index)


def segment_table(test, segments, selected):
    rows = []
    for name, part in test.assign(segment=segments, selected=selected).groupby('segment'):
        lo, hi = wilson(int(part.label.sum()), len(part))
        rows.append(dict(segment=str(name), n=len(part), share=len(part) / len(test),
                         response_rate=float(part.label.mean()), ci_low=lo, ci_high=hi,
                         target_share=float(part.selected.mean()),
                         median_recency=float(part.recency_days.median()),
                         median_frequency=float(part.frequency.median()),
                         median_monetary=float(part.monetary.median())))
    return sorted(rows, key=lambda r: -r['response_rate'])


def review(claims, evidence):
    """규칙 기반 1차 검증기. LLM Agent 출력도 같은 claim 형식으로 받아 재사용한다."""
    findings = []
    for c in claims:
        if c['kind'] == 'causal':
            findings.append(dict(id=c['id'], severity='high', reason='관측 데이터로 캠페인의 추가 효과를 주장할 수 없음'))
        elif c['kind'] == 'numeric':
            expected = evidence.get(c.get('evidence_id'))
            if expected is None:
                findings.append(dict(id=c['id'], severity='high', reason='근거 ID가 DB에 없음'))
            elif abs(c['value'] - expected) > 1e-6:
                findings.append(dict(id=c['id'], severity='medium', reason=f'수치 불일치: DB={expected:.6f}'))
    return findings


SCHEMA = '''
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, created TEXT, data_sha256 TEXT, config TEXT);
CREATE TABLE IF NOT EXISTS quality_checks(run_id TEXT, check_name TEXT, dimension TEXT, rule TEXT, violations INTEGER, share REAL, action TEXT, PRIMARY KEY(run_id, check_name));
CREATE TABLE IF NOT EXISTS metrics(evidence_id TEXT, run_id TEXT, condition TEXT, scope TEXT, subject TEXT, metric TEXT, value REAL, n INTEGER, PRIMARY KEY(run_id, evidence_id));
CREATE TABLE IF NOT EXISTS artifacts(run_id TEXT, stage TEXT, payload TEXT, PRIMARY KEY(run_id, stage));
CREATE TABLE IF NOT EXISTS agent_outputs(agent_run_id TEXT PRIMARY KEY, run_id TEXT, role TEXT, round INTEGER, model TEXT, prompt TEXT, payload TEXT, created TEXT, seconds REAL, cost_usd REAL);
'''



# ---------------------------------------------------------------- ISBN 검사·빈도 5분위 RFM (publisher.py)


def isbn13_valid(isbn):
    """978/979로 시작하는 13자리이고 체크 숫자가 맞는지."""
    def check(s):
        if len(s) != 13 or not s.isdigit() or s[:3] not in ('978', '979'):
            return False
        total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(s[:12]))
        return (10 - total % 10) % 10 == int(s[12])
    table = {s: check(s) for s in isbn.dropna().unique()}
    return isbn.map(table).eq(True)


class AccountRFM(RFM):
    """거래처는 주문 횟수 규모가 거래처마다 크게 달라 F도 학습 구간 5분위로 나눈다."""

    def fit(self, train):
        super().fit(train)
        self.f_edges = np.quantile(train.frequency, [.2, .4, .6, .8])
        return self

    def scores(self, df):
        s = super().scores(df)
        s['F'] = 1 + np.searchsorted(self.f_edges, df.frequency, side='right')
        return s



# ---------------------------------------------------------------- 보고서 틀 (report.py)


STRATEGY = {'rfm_rule': 'RFM 규칙', 'logistic': '로지스틱 회귀', 'boosting': '그래디언트 부스팅', 'random_30seeds': '무작위 (30회 평균)'}
COLORS = {'rfm_rule': '#b7791f', 'logistic': '#2b6cb0', 'boosting': '#227b70', 'random_30seeds': '#8a94a0'}
CSS = '''body{font-family:"Malgun Gothic",Arial,sans-serif;background:#f4f5f1;color:#152b32;max-width:1080px;margin:40px auto;padding:0 20px;line-height:1.7}
h1{font-size:32px;line-height:1.4}h2{margin-top:0}section{background:#fff;padding:24px 28px;border-radius:12px;margin:20px 0}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 10px;border-bottom:1px solid #e3e7e1;text-align:left}th{background:#17354b;color:#fff}
td.num{text-align:right;font-variant-numeric:tabular-nums}th,td.num{white-space:nowrap}td{min-width:84px}section{overflow-x:auto}.cards{display:flex;gap:16px;flex-wrap:wrap}.card{background:#fff;padding:18px 22px;border-radius:12px;flex:1;min-width:180px}
.card b{display:block;font-size:26px}.tag{color:#276b60;font-weight:bold}.note{color:#5b6b70;font-size:14px}.win{font-weight:bold}
.barrow{display:flex;align-items:center;gap:12px;margin:12px 0}.barrow span{width:190px}.track{flex:1;background:#eef0eb;height:22px;position:relative}.bar{height:100%}.barrow b{width:150px;font-size:14px}'''


def pct(v, digits=1):
    return '–' if v is None else f'{v * 100:.{digits}f}%'


def table(rows, columns):
    head = ''.join(f'<th>{html.escape(label)}</th>' for _, label, _ in columns)
    body = ''
    for r in rows:
        cells = ''
        for key, _, fmt in columns:
            v = r.get(key)
            text = fmt(v) if fmt else ('' if v is None else str(v))
            cells += f'<td class="{"num" if fmt else ""}">{html.escape(text)}</td>'
        body += f'<tr class="{"win" if r.get("selected_on_validation") else ""}">{cells}</tr>'
    return f'<table><tr>{head}</tr>{body}</table>'


def gains_svg(gains):
    w, h, pad = 640, 320, 48
    fx = lambda f: pad + (w - 2 * pad) * f / 0.5
    fy = lambda r: h - pad - (h - 2 * pad) * r
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="누적 이득 곡선">']
    for r in (0, .25, .5, .75, 1):
        parts.append(f'<line x1="{pad}" x2="{w - pad}" y1="{fy(r)}" y2="{fy(r)}" stroke="#e3e7e1"/><text x="{pad - 8}" y="{fy(r) + 4}" text-anchor="end" font-size="12">{r:.0%}</text>')
    for f in (.1, .2, .3, .4, .5):
        parts.append(f'<text x="{fx(f)}" y="{h - pad + 18}" text-anchor="middle" font-size="12">{f:.0%}</text>')
    parts.append(f'<line x1="{fx(.2)}" x2="{fx(.2)}" y1="{pad}" y2="{h - pad}" stroke="#c53030" stroke-dasharray="4"/>')
    parts.append(f'<polyline fill="none" stroke="#8a94a0" stroke-dasharray="3" points="{fx(0)},{fy(0)} {fx(.5)},{fy(.5)}"/>')
    for i, strategy in enumerate(['rfm_rule', 'logistic', 'boosting']):
        pts = [(0, 0)] + [(g['fraction'], g['recall']) for g in gains if g['strategy'] == strategy]
        parts.append(f'<polyline fill="none" stroke="{COLORS[strategy]}" stroke-width="2.5" points="' + ' '.join(f'{fx(a)},{fy(b)}' for a, b in pts) + '"/>')
        parts.append(f'<text x="{pad + 10}" y="{pad + 16 * i}" font-size="13" fill="{COLORS[strategy]}">■ {STRATEGY[strategy]}</text>')
    parts.append(f'<text x="{w / 2}" y="{h - 6}" text-anchor="middle" font-size="12">접촉 비율 (점선: 무작위, 빨강: 20% 제약)</text></svg>')
    return ''.join(parts)
