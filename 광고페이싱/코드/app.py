"""페이싱 시뮬레이터 데모. 실행: python -m streamlit run app.py

환경과 정책은 sim.py를 그대로 쓴다. 화면은 값을 바꿔 다시 돌려 보는 용도이고, 보고용 수치는 experiment.py로 낸다.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from experiment import DEV, FINAL, MARGIN
from sim import Asap, Config, EvenHourly, Feedback, HOURS, Scenario, make_world, metrics, simulate, summarize, target_curve

NAME = {'asap': '소진까지 집행', 'even': '시간대별 동일 할당', 'fb': '목표 곡선 추종', 'fb_v2': '추종 + 지연 보정'}
COLOR = {'소진까지 집행': '#2a78d6', '시간대별 동일 할당': '#eb6834', '목표 곡선 추종': '#1baf7a', '추종 + 지연 보정': '#eda100', '목표 곡선': '#8a8984'}
SCENARIOS = {s.label: s for s in DEV + FINAL}
LAGS = {'없음': 0, '15분': 1, '1시간': 4, '2시간': 8}
BUDGETS = {'하루 100만원': 1_000_000, '하루 3,333원 (월 10만원)': 100_000 / 30}

st.set_page_config(page_title='광고 예산 페이싱 시뮬레이터', layout='wide')
st.title('광고 예산 페이싱 시뮬레이터')
st.warning('가상 환경에서의 정책 비교입니다. 유입·가격·전환율은 모두 가정값이며, 실제 광고 효율이나 매출 개선을 뜻하지 않습니다.')

with st.sidebar:
    st.header('상황')
    budget_label = st.selectbox('예산', list(BUDGETS))
    scenario_label = st.selectbox('시나리오', list(SCENARIOS) + ['직접 설정'])
    if scenario_label == '직접 설정':
        hours = st.slider('교란 시간대', 0, 24, (11, 14))
        traffic_k = st.slider('유입 배수', 0.1, 3.0, 2.0, 0.1)
        cvr_k = st.slider('전환율 배수', 0.5, 3.0, 1.0, 0.1)
        sc = Scenario('custom', '직접 설정', traffic=((hours[0], hours[1], traffic_k),), cvr=((hours[0], hours[1], cvr_k),), lag=1)
    else:
        sc = SCENARIOS[scenario_label]
    default_lag = next((i for i, v in enumerate(LAGS.values()) if v == sc.lag), 1)
    lag = LAGS[st.selectbox('지출 보고 지연', list(LAGS), index=default_lag)]

    st.header('제어')
    kp = st.select_slider('목표 곡선 추종 Kp', [0.5, 1, 2, 4, 8, 16, 32], value=8)
    kp2 = st.select_slider('지연 보정 Kp', [0.5, 1, 2, 4, 8, 16, 32], value=16)
    ki2 = st.select_slider('지연 보정 Ki', [0.0, 0.02, 0.1, 0.5], value=0.5)

    st.header('가정값')
    cpc = st.slider('기회당 가격 중앙값(원)', 50, 500, 200, 10)
    cvr = st.slider('기준 전환율(%)', 0.5, 8.0, 2.0, 0.5) / 100
    supply = st.slider('공급 여유(전 기회 지출 ÷ 예산)', 1.2, 5.0, 2.5, 0.1)

    st.header('반복')
    n_seeds = st.slider('시드 수(표)', 10, 200, 40, 10)
    seed = st.number_input('그래프에 그릴 시드', 0, 9999, 1000)

cfg = Config(budget=BUDGETS[budget_label], cpc=cpc, cvr=cvr, supply_ratio=supply)


def policies():
    a0 = 1 / cfg.supply_ratio
    return {'asap': Asap(), 'even': EvenHourly(), 'fb': Feedback(kp, 0.0, a0=a0), 'fb_v2': Feedback(kp2, ki2, a0=a0, compensate='predict')}


@st.cache_data(show_spinner='시드별로 돌리는 중')
def table(cfg_key, sc_key, lag, kp, kp2, ki2, n):
    rows = []
    for s in range(1000, 1000 + n):
        world = make_world(cfg, sc, s)
        for name, policy in policies().items():
            rows.append(dict(scenario=sc.name, seed=s, policy=name, **metrics(simulate(policy, world, cfg.budget, lag), sc.onset)))
    return summarize(pd.DataFrame(rows))


summary = table(cfg, sc, lag, kp, kp2, ki2, n_seeds).reset_index().set_index('policy')
world = make_world(cfg, sc, int(seed))
day = {name: simulate(policy, world, cfg.budget, lag) for name, policy in policies().items()}

# 정책별 지표
shown = pd.DataFrame({
    '정책': [NAME[k] for k in summary.index],
    '소진율': [f'{v:.1%}' for v in summary.burn],
    '목표 오차': [f'{v:.1%}' for v in summary.tracking],
    '최대 오차': [f'{v:.1%}' for v in summary.peak_err],
    '전환 수/일': [f'{v:.2f}' if cfg.budget < 10_000 else f'{v:.1f}' for v in summary.conv],
    'CPA': [f'{v:,.0f}원' for v in summary.cpa],
    '95% 소진 시각': [f'{v:.1f}시' if v == v else '도달 못함' for v in summary.exhaust_h],
    '참여 확률 변동': [f'{v:.3f}' for v in summary.volatility],
    '포화 구간': [f'{v:.0%}' for v in summary.saturated],
})
st.subheader(f'{sc.label} · 시드 {n_seeds}개 평균')
st.dataframe(shown, hide_index=True, width='stretch')
st.caption(f'CPA는 총지출 ÷ 총전환입니다. 본전 기준 참고값은 교보 분석의 권당 공헌이익 {MARGIN:,}원이고, 현재 가정의 평균 가격으로 본전이 되려면 전환율 {cfg.cpc * np.exp(cfg.price_sigma ** 2 / 2) / MARGIN:.1%}가 필요합니다.')

# 해석 메모: 이 조건에서 드러난 문제를 지표 기준으로 짚는다
notes = []
fb = summary.loc['fb']
if fb.saturated > 0.15:
    notes.append(f'목표 곡선 추종이 하루의 {fb.saturated:.0%} 동안 참여 0 또는 1에 붙어 있습니다. 제어가 과하게 반응하는 신호입니다. Kp를 낮추거나 지연 보정을 비교해 보세요.')
if summary.loc['even'].burn < 0.5:
    notes.append(f'시간대별 동일 할당의 소진율이 {summary.loc["even"].burn:.0%}입니다. 시간당 예산({cfg.budget / 24:,.0f}원)이 기회 가격보다 작으면 값싼 기회만 사게 됩니다. CPA가 낮아 보여도 전환 수를 함께 보세요.')
if summary.loc['fb'].burn < 0.97:
    notes.append(f'목표 곡선 추종의 소진율이 {fb.burn:.1%}입니다. 유입이 예측보다 적거나 예산이 작으면 제어가 뒤처진 채 하루가 끝납니다.')
if cfg.budget < 10_000:
    notes.append('이 예산에서는 하루 전환이 1건도 안 됩니다. 시드 평균의 차이는 커 보여도 실제 집행으로 확인하기는 어렵습니다.')
for n in notes:
    st.info(n)

# 그래프 공통
frames = [pd.DataFrame(dict(시각=HOURS + .25, 값=target_curve(1.0) * 100, 정책='목표 곡선'))]
frames += [pd.DataFrame(dict(시각=HOURS + .25, 값=r.spend / cfg.budget * 100, 정책=NAME[k])) for k, r in day.items()]
spend = pd.concat(frames)
x = alt.X('시각:Q', scale=alt.Scale(domain=[0, 24]), axis=alt.Axis(values=list(range(0, 25, 6)), title='시'))
hover = alt.selection_point(fields=['시각'], nearest=True, on='pointerover', empty=False)


def line_chart(data, y_title, domain=None):
    shown = [k for k in COLOR if k in set(data['정책'])]  # 정책별 색은 고정하고, 범례에는 그래프에 있는 정책만 넣는다
    color_scale = alt.Scale(domain=shown, range=[COLOR[k] for k in shown])
    y = alt.Y('값:Q', title=y_title, scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    line = alt.Chart(data).mark_line(strokeWidth=2).encode(x=x, y=y, color=alt.Color('정책:N', scale=color_scale, legend=alt.Legend(orient='top', title=None, columns=3)))
    tips = alt.Chart(data).mark_circle(size=70).encode(
        x=x, y=y, color=alt.Color('정책:N', scale=color_scale, legend=None),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip('시각:Q', format='.2f'), '정책:N', alt.Tooltip('값:Q', format='.2f')]).add_params(hover)
    return (line + tips).properties(height=320)


left, right = st.columns(2)
with left:
    st.subheader('누적 지출')
    st.altair_chart(line_chart(spend, '예산 대비 %', [0, 105]), width='stretch')
with right:
    st.subheader('누적 전환')
    conv = pd.concat([pd.DataFrame(dict(시각=HOURS + .25, 값=r.conv, 정책=NAME[k])) for k, r in day.items()])
    st.altair_chart(line_chart(conv, '전환 수'), width='stretch')

left, right = st.columns(2)
with left:
    st.subheader('참여 확률')
    a = pd.concat([pd.DataFrame(dict(시각=HOURS, 값=day[k].a, 정책=NAME[k])) for k in ('fb', 'fb_v2')])
    st.altair_chart(line_chart(a, '참여 확률', [0, 1]), width='stretch')
with right:
    st.subheader('목표 대비 지출 오차')
    target = target_curve(cfg.budget)
    err = pd.concat([pd.DataFrame(dict(시각=HOURS + .25, 값=(r.spend - target) / cfg.budget * 100, 정책=NAME[k])) for k, r in day.items()])
    st.altair_chart(line_chart(err, '예산 대비 %'), width='stretch')

st.caption(f'그래프는 시드 {int(seed)} 하루입니다. 표는 시드 평균이라 값이 다를 수 있습니다.')
