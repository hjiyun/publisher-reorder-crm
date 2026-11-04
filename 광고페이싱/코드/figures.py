"""README 그림 4종을 최근 final·dev 실행 결과로 만든다. 실행: python figures.py  → ../그림/*.png"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sim
from experiment import DEV_SEEDS, FINAL, RESULTS, LARGE, frozen_policies
from sim import HOURS, make_world, simulate, target_curve

OUT = RESULTS.parent / '그림'
OUT.mkdir(exist_ok=True)

# 정책별 색은 고정한다(검증 팔레트 1~4번). 목표 곡선은 회색 점선
COLOR = {'asap': '#2a78d6', 'even': '#eb6834', 'fb': '#1baf7a', 'fb_v2': '#eda100'}
NAME = {'asap': '소진까지 집행', 'even': '시간대별 동일 할당', 'fb': '목표 곡선 추종', 'fb_v2': '추종 + 지연 보정'}
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0', '#fcfcfb'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold'})
NOTE = '가상 환경에서의 정책 비교. 실제 광고 성과가 아니다.'


def runs(stage):
    found = [p.parent for p in sorted(RESULTS.glob('*/manifest.json')) if json.loads(p.read_text(encoding='utf8'))['stage'] == stage]
    return found[-1]


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches='tight', facecolor=SURF)
    plt.close(fig)
    print(name)


def hour_axis(ax):
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xticklabels([f'{h}시' for h in range(0, 25, 6)])
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def mean_curves(dev, sc, seeds):
    curves = {name: [] for name in NAME}
    for seed in seeds:
        world = make_world(LARGE, sc, seed)
        for name, policy in frozen_policies(dev).items():
            curves[name].append(simulate(policy, world, LARGE.budget, sc.lag).spend / LARGE.budget)
    return {k: np.mean(v, axis=0) for k, v in curves.items()}


def fig_curves(dev):
    """누적 지출과 목표 곡선. 교란 구간을 함께 표시한다."""
    picks = ['surge_late', 'evening_cvr_price', 'drop_morning', 'surge_lag']
    scenarios = {s.name: s for s in FINAL}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    target = target_curve(1.0)
    for ax, key in zip(axes, picks):
        sc = scenarios[key]
        for start, end, _ in sc.traffic + sc.cvr:
            ax.axvspan(start, end, color='#f0efec', lw=0, zorder=0)
        ax.plot(HOURS + .25, target, color=MUTED, lw=1.6, ls=(0, (4, 3)), label='목표 곡선')
        for name, curve in mean_curves(dev, sc, range(1000, 1060)).items():
            ax.plot(HOURS + .25, curve, color=COLOR[name], lw=2, label=NAME[name])
        ax.set_title(sc.label, loc='left')
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        hour_axis(ax)
    axes[0].set_ylabel('예산 대비 누적 지출')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(.5, 1.08))
    fig.text(0, -0.03, f'{NOTE} 시드 60개 평균, 하루 예산 100만원. 음영은 교란 구간.', color=MUTED, fontsize=9.5)
    fig.tight_layout()
    save(fig, '1_누적지출.png')


def fig_lift(final):
    """소진까지 집행 대비 전환 수 차이. 페이싱의 이득이 시간대 가정에 달려 있음을 보여 준다."""
    d = pd.read_csv(final / 'final_diff_large.csv')
    base = pd.read_csv(final / 'final_summary_large.csv')
    base = base[base.policy == 'asap'].set_index('scenario').conv
    labels = {s.name: s.label for s in FINAL}
    order = ['evening_cvr_price', 'surge_then_drop', 'normal', 'surge_late', 'drop_morning', 'surge_lag', 'drop_lag', 'flat', 'low_day', 'morning_cvr']
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, key in enumerate(order):
        for name, off in (('fb', -.15), ('fb_v2', .15)):
            r = d[(d.scenario == key) & (d.policy == name)].iloc[0]
            y = len(order) - 1 - i + off
            ax.plot([r.low / base[key] * 100, r.high / base[key] * 100], [y, y], color=COLOR[name], lw=2, solid_capstyle='round')
            ax.plot(r['diff'] / base[key] * 100, y, 'o', color=COLOR[name], ms=7, mec=SURF, mew=1.5, label=NAME[name] if i == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[k] for k in order[::-1]])
    ax.set_xlabel('소진까지 집행 대비 전환 수 차이(%), 선은 95% 구간')
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, loc='lower right')
    ax.set_title('페이싱의 이득은 전환이 몰리는 시간대 가정에서 나온다', loc='left')
    fig.text(0, -0.04, f'{NOTE} 평가 전용 시나리오, 시드 200개, 같은 시드끼리 비교.', color=MUTED, fontsize=9.5)
    fig.tight_layout()
    save(fig, '2_전환차이.png')


def fig_lag_map(dev_dir):
    """제어 강도 × 보고 지연별 목표 오차. 튜닝 시나리오 평균."""
    d = pd.read_csv(dev_dir / 'dev_lag.csv')
    d = d[d.lag.isin([0, 1, 4, 8])]
    titles = {'none': '보정 없음(처음 구현)', 'predict': '지연 보정(미보고 지출 추정)'}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    vmax = d.tracking.max() * 100
    for ax, key in zip(axes, titles):
        m = d[d.compensate == key].pivot(index='lag', columns='kp', values='tracking') * 100
        im = ax.imshow(m.values, cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels([f'{k:g}' for k in m.columns])
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels({0: '없음', 1: '15분', 4: '1시간', 8: '2시간'}[l] for l in m.index)
        ax.set_xlabel('제어 강도 Kp')
        ax.set_title(titles[key], loc='left')
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=10, color='white' if v > vmax * .55 else INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel('지출 보고 지연')
    fig.text(0, -0.05, '칸의 값은 목표 곡선과의 평균 오차(예산 대비 %). 낮을수록 좋다. 튜닝 시나리오 4개 평균, 시드 40개.', color=MUTED, fontsize=9.5)
    fig.tight_layout()
    save(fig, '3_지연과제어강도.png')


def fig_failure(dev):
    """보고 2시간 지연 + 급증에서 참여 확률과 목표 오차. 처음 구현이 과하게 반응하고 포화되는 모습."""
    sc = {s.name: s for s in FINAL}['surge_lag']
    seed = 1000
    world = make_world(LARGE, sc, seed)
    policies = frozen_policies(dev)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.4), sharex=True, gridspec_kw=dict(hspace=.28))
    target = target_curve(LARGE.budget)
    for ax in axes:
        for start, end, _ in sc.traffic:
            ax.axvspan(start, end, color='#f0efec', lw=0, zorder=0)
    for name in ('fb', 'fb_v2'):
        r = simulate(policies[name], world, LARGE.budget, sc.lag)
        axes[0].step(HOURS, r.a, where='post', color=COLOR[name], lw=2, label=NAME[name])
        axes[1].plot(HOURS + .25, (r.spend - target) / LARGE.budget * 100, color=COLOR[name], lw=2)
    axes[0].set_title('처음 구현은 뒤늦은 보고에 반응해 참여를 끝까지 올렸다 내린다', loc='left')
    axes[0].set_ylabel('참여 확률')
    axes[0].set_yticks([0, .5, 1])
    axes[1].legend(*axes[0].get_legend_handles_labels(), frameon=False, loc='upper left', ncol=2)
    axes[1].axhline(0, color=INK2, lw=1)
    axes[1].set_ylabel('목표 대비 지출 오차(%)')
    hour_axis(axes[0])
    hour_axis(axes[1])
    fig.text(0.02, 0.0, f'{NOTE} 시드 1개의 하루, 하루 예산 100만원. 음영은 트래픽 2배 구간(11~14시).', color=MUTED, fontsize=9.5)
    save(fig, '4_실패사례_지연.png')


if __name__ == '__main__':
    final, dev_dir = runs('final'), runs('dev')
    dev = json.loads((dev_dir / 'manifest.json').read_text(encoding='utf8'))
    fig_curves(dev)
    fig_lift(final)
    fig_lag_map(dev_dir)
    fig_failure(dev)
