"""발표 그림 9종을 저장된 결과 파일에서 다시 만든다. 실행: python make_figures.py  → 같은 폴더에 PNG

수치 출처
- 예측 성능: 교보/결과/20261004-222853-e593d5, 예스24/결과/20261004-223032-57cd3b (02_models.json, full 조건)
- 데이터 정제: 같은 실행의 01_quality.json, 오류실험 탐지율
- 오류 비율별 성능: 오류실험/결과/20261004-225800-66e06d/results.json (부스팅, 시드 10개)
- 에이전트: 에이전트/결과/개선/20261006-115556-0eb3a6 (롤링 검증 실행), 20261006-131103-08c501 (도서 지식 실행)
- 독자 구성·광고 손익분기: 각 서점 04_crm_roi.json
- 선정 도서: 20261006-131103-08c501 의 recommendations(최종안) + 기준일 2026-10-01 스냅숏
"""
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
BASE = OUT.parent.parent  # 마린북스/
RUNS = {'교보': BASE / '교보/결과/20261004-222853-e593d5', '예스24': BASE / '예스24/결과/20261004-223032-57cd3b'}
IMPROVE = BASE / '에이전트/결과/개선/20261006-115556-0eb3a6'
KNOW = BASE / '에이전트/결과/개선/20261006-131103-08c501'
ERR = BASE / '오류실험/결과/20261004-225800-66e06d/results.json'

# 팔레트(검증 통과: 파랑·주황·청록 순서 고정), 잉크·면
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#a3a29c'
INK, INK2, MUTED, GRID, SURF, PANEL = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0', '#fcfcfb', '#f0efec'
STORE = {'교보': BLUE, '예스24': ORANGE}
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 12})
load = lambda p: json.load(open(p, encoding='utf8'))


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches='tight', facecolor=SURF)
    plt.close(fig)
    print(name)


def rounded_bar(ax, x, h, w, color):
    """끝이 둥근(4px 상당) 막대. 기준선 쪽은 각지게."""
    r = min(0.012, h / 2)
    ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=color, ec=SURF, lw=1.5, mutation_aspect=1 / 3))
    ax.add_patch(plt.Rectangle((x - w / 2, 0), w, min(h, r * 2), fc=color, ec='none'))


# ---------------------------------------------------------------- 1. 예측 성능 비교
def fig_performance():
    names = [('random_30seeds', '무작위', GRAY), ('rfm_rule', '최근 판매 규칙', ORANGE), ('model', '예측 모델', BLUE)]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    w = 0.24
    for i, (store, run) in enumerate(RUNS.items()):
        res = [r for r in load(run / '02_models.json')['results'] if r['condition'] == 'full']
        val = {r['strategy']: r['precision20'] for r in res}
        val['model'] = next(r['precision20'] for r in res if r['selected_on_validation'])
        for j, (k, label, color) in enumerate(names):
            x = i + (j - 1) * (w + 0.03)
            ax.bar(x, val[k], w, color=color, edgecolor=SURF, linewidth=2)
            ax.text(x, val[k] + 0.012, f'{val[k] * 100:.0f}%', ha='center', va='bottom', fontsize=13, color=INK, fontweight='bold' if k == 'model' else 'normal')
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 0.72)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['교보 (재주문 = 입하)', '예스24 (재주문 = 발주)'], fontsize=13, color=INK); ax.set_yticks([0, .2, .4, .6]); ax.set_yticklabels(['0%', '20%', '40%', '60%'])
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.spines['left'].set_visible(False); ax.tick_params(length=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c) for _, _, c in names], labels=[n for _, n, _ in names], loc='upper left', ncol=3, frameon=False, fontsize=12)
    ax.set_title('다음 30일 재주문 도서 상위 20% 선정 정확도 (Precision@20%)', loc='left', fontsize=15, pad=14)
    fig.text(0.125, -0.03, '시험 구간: 마지막 6개월(매월 1일마다 상위 20% 선정 후 합산). 무작위는 30회 평균.', fontsize=10, color=MUTED)
    save(fig, '01_예측성능_비교.png')


# ---------------------------------------------------------------- 2. 전체 시스템 흐름도
def box(ax, x, y, w, h, title, body='', fc=PANEL, ec=GRID, tc=INK, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 0.22, title, ha='center', va='top', fontsize=13, fontweight='bold', color=tc)
    if body:
        ax.text(x + w / 2, y + h - 0.62, body, ha='center', va='top', fontsize=10.5, color=INK2, linespacing=1.45)


def arrow(ax, a, b, color=INK2, style='-|>', lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=16, color=color, lw=lw, connectionstyle=f'arc3,rad={rad}'))


def fig_system():
    fig, ax = plt.subplots(figsize=(15, 5.0)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.0); ax.axis('off')
    steps = [('서점 SCM 자료', '교보문고 · 예스24\n입하/발주 · 반품 · 월별 판매\n도서정보 · 구매자 집계'),
             ('SQL · 품질 점검', 'SQLite 적재\n점검 규칙 17개\n(수량 0·중복·ISBN·날짜)'),
             ('변수 생성', '기준일(매월 1일) 이전 기록만\n입하·반품·판매 합계·횟수\n최근성 · 판매 추세'),
             ('예측', '30일 재주문 확률\n무작위 · 판매 규칙\n로지스틱 · 부스팅'),
             ('도서 선정 · 보고서', '매월 상위 20% 선정\nHTML 보고서\nROI · CRM 보조 분석')]
    w, h, gap, y = 2.55, 1.85, 0.38, 2.45
    for i, (t, b) in enumerate(steps):
        x = 0.25 + i * (w + gap)
        box(ax, x, y, w, h, t, b, fc='#e8f0fb' if i in (3, 4) else PANEL, ec=BLUE if i in (3, 4) else GRID)
        if i:
            arrow(ax, (x - gap + 0.02, y + h / 2), (x - 0.02, y + h / 2))
    box(ax, 3.18, 0.2, 8.6, 1.35, '에이전트 검토 · 개선 공정 (공정 DB)', '분석가 A·B 독립 제안 → 조정 · 토론(수정 2회) → 적대적 검토 → 최종안\n'
        '에이전트는 집계 결과만 읽고(읽기 전용 SQL), 모든 중간 산출물을 DB에 저장', fc='#fdf0ea', ec=ORANGE)
    arrow(ax, (0.25 + 1 * (w + gap) + w / 2, y), (4.9, 1.57), color=ORANGE, rad=0.15)
    arrow(ax, (10.1, 1.57), (0.25 + 3 * (w + gap) + w / 2, y), color=ORANGE, rad=0.15)
    ax.text(0.25, 4.95, '전체 시스템 흐름', fontsize=16, fontweight='bold', va='top')
    save(fig, '02_시스템_흐름도.png')


# ---------------------------------------------------------------- 3. 학습·검증·시험 기간
def fig_periods():
    fig = plt.figure(figsize=(15, 6.4))
    ax = fig.add_axes([0.06, 0.42, 0.9, 0.42])
    to_n = lambda s: (pd.Timestamp(s).year - 2022) * 12 + pd.Timestamp(s).month - 1
    rows = []
    for k, (store, run) in enumerate(RUNS.items()):
        c = load(run / '01_quality.json')['config']['cutoffs']
        rows.append((f'{store}', [('학습', c['train'][0], c['train'][-1], GRAY), ('검증', c['val'][0], c['val'][-1], ORANGE), ('시험', c['test'][0], c['test'][-1], BLUE)]))
    pre = None
    for k, (label, parts) in enumerate(rows):
        yy = 1.6 - k * 0.9
        for name, a, b, col in parts:
            x0, x1 = to_n(a), to_n(b) + 1
            ax.add_patch(FancyBboxPatch((x0, yy), x1 - x0 - 0.15, 0.55, boxstyle='round,pad=0,rounding_size=0.12', fc=col, ec=SURF, lw=2, mutation_aspect=0.2))
            ax.text((x0 + x1) / 2, yy + 0.27, f'{name}\n{a[:7]}\n~ {b[:7]}' if name != '학습' else f'{name} ({a[:7]} ~ {b[:7]}, 기준일 {x1 - x0}개)',
                    ha='center', va='center', fontsize=10.5 if name == '학습' else 9.5, linespacing=1.15, color='white' if name != '학습' else INK, fontweight='bold')
        ax.text(-0.8, yy + 0.27, label, ha='right', va='center', fontsize=13, fontweight='bold')
    ax.set_xlim(0, to_n('2026-12-01')); ax.set_ylim(0.4, 2.3); ax.axis('off')
    for yr in range(2022, 2027):
        ax.text(to_n(f'{yr}-01-01'), 2.25, str(yr), fontsize=10.5, color=MUTED, ha='left')
        ax.axvline(to_n(f'{yr}-01-01'), color=GRID, lw=0.8, zorder=0)
    fig.text(0.06, 0.93, '기간 나누기: 과거로 학습하고, 이후 구간에서 평가', fontsize=16, fontweight='bold')
    # 기준일 하나의 구조
    ax2 = fig.add_axes([0.06, 0.02, 0.9, 0.3]); ax2.set_xlim(0, 15); ax2.set_ylim(0, 3); ax2.axis('off')
    ax2.add_patch(FancyBboxPatch((0.3, 1.1), 7.6, 0.8, boxstyle='round,pad=0,rounding_size=0.15', fc=PANEL, ec=GRID))
    ax2.text(4.1, 1.5, '기준일 이전 기록으로 변수 계산 (입하·반품 30/90/365일, 판매 1/3/12개월, 최근성)', ha='center', va='center', fontsize=11.5)
    ax2.add_patch(FancyBboxPatch((8.25, 1.1), 3.2, 0.8, boxstyle='round,pad=0,rounding_size=0.15', fc='#e8f0fb', ec=BLUE))
    ax2.text(9.85, 1.5, '다음 30일: 재주문 1건 이상?\n= 정답(라벨)', ha='center', va='center', fontsize=11.5, color=INK)
    ax2.plot([8.08, 8.08], [0.8, 2.25], color=ORANGE, lw=2.5)
    ax2.text(8.08, 2.45, '기준일 T (매월 1일)', ha='center', fontsize=12, color=ORANGE, fontweight='bold')
    ax2.text(11.8, 1.5, '→ 매월 상위 20% 선정 후\n    실제 재주문 여부로 채점', va='center', fontsize=11.5, color=INK2)
    ax2.text(0.3, 0.45, '에이전트 개선 공정은 시험 직전 18개월을 6개월씩 3구간으로 나눠 롤링 검증한다(구간마다 그 이전 자료로만 학습). 시험 구간은 마지막에 한 번만 쓴다.',
             fontsize=10.5, color=MUTED)
    save(fig, '03_기간_나누기.png')


# ---------------------------------------------------------------- 4. 선정 도서 표
def fig_selection():
    sys.path.insert(0, str(BASE / '교보/코드'))
    sys.path.insert(0, str(BASE / '오류실험/코드'))
    import error_injection as ei
    import kyobo
    for store in ('예스24', '교보'):
        db = sqlite3.connect(KNOW / f'process_{store}.sqlite')
        rec = pd.read_sql('SELECT rank, isbn, title, score, cutoff FROM recommendations ORDER BY rank', db)
        choice = db.execute('SELECT choice FROM review').fetchone()[0]
        raw = ei.STORES[store]['load']()[:4]
        _, drop = kyobo.quality(*raw)
        books, r, t, s = kyobo.clean(*raw, drop, True)
        cut = pd.Timestamp(rec.cutoff.iloc[0])
        snap = kyobo.snapshot(books, r, t, s, cut).set_index('isbn')
        rec = rec.join(snap[['sale_1m', 'sale_3m', 'sale_12m', 'days_since_rcvd', 'rcvd_n_365d']], on='isbn')
        rows = rec.head(15)
        fig, ax = plt.subplots(figsize=(13, 0.42 * (len(rows) + 1.6) + 0.9)); ax.axis('off')
        cols = ['순위', '도서', '재주문\n확률 점수', '지난달\n판매', '최근 3개월\n판매', '최근 12개월\n판매', '직전 재주문\n후 경과일', '최근 1년\n재주문 횟수']
        short = lambda t: t if len(t) <= 24 else t[:23] + '…'
        cells = [[f'{r.rank}', short(r.title), f'{r.score:.2f}', f'{r.sale_1m:.0f}권', f'{r.sale_3m:.0f}권', f'{r.sale_12m:.0f}권',
                  '–' if pd.isna(r.days_since_rcvd) else f'{r.days_since_rcvd:.0f}일', f'{r.rcvd_n_365d:.0f}회'] for r in rows.itertuples()]
        tb = ax.table(cellText=cells, colLabels=cols, cellLoc='center', colWidths=[.05, .35, .09, .08, .1, .1, .12, .11], bbox=[0, 0, 1, 1])
        tb.auto_set_font_size(False); tb.set_fontsize(10.5)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor(GRID)
            if i == 0:
                c.set_facecolor('#17354b'); c.set_text_props(color='white', fontweight='bold')
            else:
                c.set_facecolor(SURF if i % 2 else PANEL)
                if j == 1:
                    c.set_text_props(ha='left'); c._loc = 'left'
        kind = '에이전트 공정 최종안' if choice != 'baseline' else '기준 모델(에이전트 검토에서 유지)'
        ax.set_title(f'{store} · {rec.cutoff.iloc[0][:7]} 재주문 예상 도서 (상위 20% {len(rec)}종 중 상위 {len(rows)}종)', loc='left', fontsize=15, pad=8)
        fig.text(0.125, 0.0, f'모델: {kind}. 점수는 다음 30일 안에 재주문될 확률 추정치. 판매·재주문은 기준일 이전 기록.', fontsize=10, color=MUTED)
        save(fig, f'04_선정도서_{store}.png')


# ---------------------------------------------------------------- 5. 데이터 정제 사례표
def fig_cleaning():
    q = {s: {(x['table'], x['check']): x for x in load(run / '01_quality.json')['quality']} for s, run in RUNS.items()}
    det = {(d['table'], d['type']): d['full'] for d in load(ERR)['detection'] if d['store'] == '교보'}
    rows = [('수량 0 이하 발주·입하', ('입하', 'qty_nonpositive'), '제외: 재주문으로 잘못 세지 않도록', det[('rcvd', 'qty_zero')]),
            ('완전 중복 행', ('입하', 'exact_duplicate'), '제외: 같은 기록이 두 번 들어간 경우', det[('rcvd', 'duplicate')]),
            ('ISBN 체크 숫자 오류', ('입하', 'isbn_invalid'), '제외: 도서와 연결할 수 없음', det[('rcvd', 'isbn_typo')]),
            ('도서정보에 없는 ISBN', ('입하', 'isbn_not_in_books'), '제외', None),
            ('출간일 이전 입하', ('입하', 'before_publication'), '기록만: 예약 입하일 수 있음', None),
            ('누적 반품 > 누적 입하', ('반품', 'return_exceeds_received'), '기록만: 기록 시작 전 공급분 가능', None),
            ('판매 수량 음수', ('판매', 'negative_sales'), '기록만: 반품 상계', None),
            ('출판일 누락 도서', ('도서', 'pub_missing'), '예측 대상에서 제외', None)]
    n = lambda s, k: q[s].get(k, {}).get('violations', 0)
    cells = [[name, f'{n("교보", k):,}건', f'{n("예스24", k):,}건', act, '–' if d is None else f'{d * 100:.0f}%'] for name, k, act, d in rows]
    fig, ax = plt.subplots(figsize=(14, 4.4)); ax.axis('off')
    tb = ax.table(cellText=cells, colLabels=['점검 항목', '교보 실제 발견', '예스24 실제 발견', '처리', '오류 주입 실험 탐지율'],
                  cellLoc='center', colWidths=[.22, .13, .13, .34, .16], bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False); tb.set_fontsize(11)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_facecolor('#17354b'); c.set_text_props(color='white', fontweight='bold')
        else:
            c.set_facecolor(SURF if i % 2 else PANEL)
            if (j in (1, 2)) and cells[i - 1][j] != '0건':
                c.set_text_props(fontweight='bold', color=ORANGE if '기록만' not in cells[i - 1][3] else INK)
    ax.set_title('데이터 품질 점검: 무엇을 찾았고 어떻게 처리했나', loc='left', fontsize=15, pad=8)
    fig.text(0.125, 0.0, '예스24 발주의 수량 0 행(약 28%)은 점검이 없으면 "재주문"으로 잘못 세어진다. 탐지율은 일부러 넣은 오류를 규칙이 잡은 비율(교보, 오류 20%).',
             fontsize=10, color=MUTED)
    save(fig, '05_데이터정제_사례.png')


# ---------------------------------------------------------------- 6. 오류 비율별 성능
def fig_errors():
    rows = [r for r in load(ERR)['by_rate'] if r['strategy'] == 'boosting']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, store in zip(axes, ('교보', '예스24')):
        d = sorted([r for r in rows if r['store'] == store], key=lambda r: r['rate'])
        x = [r['rate'] * 100 for r in d]
        for key, label, color in (('minimal', '품질 점검 미적용', ORANGE), ('full', '품질 점검 적용', BLUE)):
            y = np.array([r[key] for r in d]) * 100
            sd = np.array([r[f'{key}_sd'] for r in d]) * 100
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.12, lw=0)
            ax.plot(x, y, color=color, lw=2, marker='o', ms=7, mec=SURF, mew=2, label=label)
            ax.text(x[-1] + 0.6, y[-1], f'{y[-1]:.1f}%', va='center', fontsize=11, color=INK)
        ax.set_title(store, loc='left', fontsize=13)
        ax.set_xticks([0, 5, 10, 20]); ax.set_xticklabels(['0%', '5%', '10%', '20%']); ax.set_xlim(-1, 24)
        ax.set_xlabel('일부러 넣은 오류 비율'); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    axes[0].set_ylabel('Precision@20% (시험 구간)'); axes[0].set_ylim(44, 63); axes[0].set_yticks(range(44, 64, 4))
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    axes[1].legend(loc='upper right', frameon=False, fontsize=11)
    fig.suptitle('오류가 늘면 성능이 떨어지지만, 기존 점검 규칙은 차이를 만들지 못했다', x=0.07, ha='left', fontsize=15, y=1.02)
    fig.text(0.07, -0.04, '오류 6종(수량 0·중복·ISBN 오타·단위 ×10·날짜 1년 오기·행 누락)을 같은 비율로 주입. 선 = 시드 10개 평균, 띠 = ±1 표준편차. 부스팅 모델.',
             fontsize=10, color=MUTED)
    save(fig, '06_오류비율별_성능.png')


# ---------------------------------------------------------------- 7. 에이전트 구조도 + 성능 비교표
def fig_agents():
    fig, ax = plt.subplots(figsize=(15, 4.9)); ax.set_xlim(0, 15); ax.set_ylim(0, 5.6); ax.axis('off')
    box(ax, 0.2, 1.95, 2.6, 1.7, '0. 분석 → DB', '기준 모델 성능\n변수 중요도 · 오답 구간\n도서 지식(제목 분류)')
    box(ax, 3.4, 3.2, 2.5, 1.1, '분석가 A', '변수 중심 제안', fc='#e8f0fb', ec=BLUE)
    box(ax, 3.4, 1.3, 2.5, 1.1, '분석가 B', '모델·기간·정제 제안', fc='#e8f0fb', ec=BLUE)
    box(ax, 6.5, 1.95, 2.5, 1.7, '채점', '롤링 검증 3구간\n개선 확률 · 구간별 차이\n→ DB 저장')
    box(ax, 9.6, 1.95, 2.5, 1.7, '토론 (수정 2회)', '서로의 안과 점수를 읽고\n반박 · 결합한 수정안')
    box(ax, 12.6, 1.95, 2.25, 1.7, '적대적 검토', '사전 채택 기준\n선택 편향 점검\n수치 주장 DB 대조', fc='#fdf0ea', ec=ORANGE)
    for a, b in (((2.8, 3.2), (3.4, 3.75)), ((2.8, 2.4), (3.4, 1.85)), ((5.9, 3.75), (6.5, 3.2)), ((5.9, 1.85), (6.5, 2.4)), ((9.0, 2.8), (9.6, 2.8)), ((12.1, 2.8), (12.6, 2.8))):
        arrow(ax, a, b)
    arrow(ax, (10.85, 1.93), (7.75, 1.93), color=MUTED, rad=-0.35)
    ax.text(9.3, 0.95, '재채점', ha='center', fontsize=10.5, color=MUTED)
    ax.text(0.2, 5.45, 'LLM 멀티 에이전트 개선 공정', fontsize=16, fontweight='bold', va='top')
    ax.text(0.2, 4.95, '에이전트는 원본 행·ISBN 없이 공정 DB를 읽기 전용 SQL로만 조회하고, 제안·점수·토론·검토를 모두 DB에 남긴다.', fontsize=11, color=INK2, va='top')
    save(fig, '07a_에이전트_구조도.png')

    stages = ['기준 모델', '경쟁 (라운드1 최고)', '토론 (전 라운드 검증 최고, 검토 없음)', '적대적 검토 (최종안)']
    label = {'기준 모델': '기준 모델', '경쟁 (라운드1 최고)': '경쟁 (1라운드 최고)', '토론 (전 라운드 검증 최고, 검토 없음)': '토론 (검증 최고, 검토 없음)',
             '적대적 검토 (최종안)': '적대적 검토 (최종)'}
    res = {}
    for store in ('교보', '예스24'):
        db = sqlite3.connect(IMPROVE / f'process_{store}.sqlite')
        res[store] = {r[0]: r[1:] for r in db.execute('SELECT stage, proposal_id, test_p20, test_ap FROM final_results')}
    cells = []
    for st in stages:
        k, y = res['교보'][st], res['예스24'][st]
        bk, by = res['교보']['기준 모델'], res['예스24']['기준 모델']
        fmt = lambda v, b: f'{v * 100:.1f}%' + ('' if abs(v - b) < 1e-9 else f' ({(v - b) * 100:+.1f}%p)')
        cells.append([label[st], fmt(k[1], bk[1]), fmt(y[1], by[1]), fmt(y[2], by[2])])
    fig, ax = plt.subplots(figsize=(13, 2.6)); ax.axis('off')
    tb = ax.table(cellText=cells, colLabels=['단계', '교보 Precision@20%', '예스24 Precision@20%', '예스24 AP(순위 품질)'], cellLoc='center',
                  colWidths=[.34, .22, .22, .22], bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False); tb.set_fontsize(11.5)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_facecolor('#17354b'); c.set_text_props(color='white', fontweight='bold')
        else:
            c.set_facecolor('#fdf0ea' if i == len(cells) else (SURF if i % 2 else PANEL))
            if i == len(cells):
                c.set_text_props(fontweight='bold')
    ax.set_title('단계별 시험 구간 성능 (선택은 검증 구간으로만, 시험은 마지막에 한 번)', loc='left', fontsize=14, pad=6)
    fig.text(0.125, -0.06, '롤링 검증 실행(2026-10-06). 교보는 채택 기준을 넘은 안이 없어 기준 모델 유지. 공정 수정 전 실행에서는 검증 최고안이 시험에서 -2.6%p였고 적대적 검토가 이를 거부했다.',
             fontsize=9.5, color=MUTED)
    save(fig, '07b_에이전트_성능표.png')


# ---------------------------------------------------------------- 8. 독자 구성
def fig_readers():
    p = {s: load(run / '04_crm_roi.json')['personas'] for s, run in RUNS.items()}
    ages = ['10대 이하', '20대', '30대', '40대', '50대 이상']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[2.3, 1]))
    w = 0.36
    for k, (store, d) in enumerate(p.items()):
        for i, a in enumerate(ages):
            v = d['age_share'].get(a, 0)
            ax.bar(i + (k - 0.5) * (w + 0.03), v, w, color=STORE[store], label=store if i == 0 else None, edgecolor=SURF, linewidth=1.5)
            ax.text(i + (k - 0.5) * (w + 0.03), v + 0.01, f'{v * 100:.0f}%', ha='center', va='bottom', fontsize=11)
    ax.set_xticks(range(len(ages))); ax.set_xticklabels(ages); ax.set_ylim(0, 0.75)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v * 100:.0f}%')); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0); ax.legend(frameon=False, fontsize=12, loc='upper left'); ax.set_title('구매자 연령대 (연령 확인분 기준)', loc='left', fontsize=13)
    for k, (store, d) in enumerate(p.items()):
        f = d['female_share']
        ax2.barh(k, f, color=STORE[store], height=0.5, edgecolor=SURF)
        ax2.barh(k, 1 - f, left=f, color=PANEL, height=0.5, edgecolor=SURF)
        ax2.text(f / 2, k, f'여성 {f * 100:.0f}%', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        ax2.text(1.02, k, f'남성\n{(1 - f) * 100:.0f}%', ha='left', va='center', color=INK2, fontsize=11)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(list(p)); ax2.set_xlim(0, 1.16); ax2.set_xticks([]); ax2.invert_yaxis()
    ax2.spines['left'].set_visible(False); ax2.spines['bottom'].set_visible(False); ax2.tick_params(length=0)
    ax2.set_title('성별 (성별 확인분 기준)', loc='left', fontsize=13)
    fig.suptitle('주 구매층은 40대 여성: 학부모가 사서 아이가 쓰는 책', x=0.06, ha='left', fontsize=15, y=1.02)
    fig.text(0.06, -0.04, f'기간 {p["교보"]["period"]} (교보), {p["예스24"]["period"]} (예스24). 서점 구매자 집계(개인 단위 아님). '
             '예스24는 10세 미만 명의 구매가 많다.', fontsize=10, color=MUTED)
    save(fig, '08_독자구성.png')


# ---------------------------------------------------------------- 9. 광고 손익분기 히트맵
def fig_breakeven():
    budgets = np.array([3, 5, 10, 20, 30, 50]) * 10000
    uplifts = np.array([0.05, 0.1, 0.2, 0.3, 0.5])
    cmap = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', BLUE])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for ax, (store, run) in zip(axes, RUNS.items()):
        s = next(r for r in load(run / '04_crm_roi.json')['breakeven']['series'] if r['series'] == '발자취')
        profit = s['monthly_sales'] * uplifts[:, None] * s['margin'] - budgets[None, :]
        im = ax.imshow(profit / 10000, cmap=cmap, norm=TwoSlopeNorm(0, -50, 20), aspect='auto', origin='lower')
        for i in range(len(uplifts)):
            for j in range(len(budgets)):
                v = profit[i, j] / 10000
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=11, color='white' if abs(v) > 25 else INK)
        ax.set_xticks(range(len(budgets))); ax.set_xticklabels([f'{b // 10000}만' for b in budgets]); ax.set_xlabel('월 광고비')
        ax.set_yticks(range(len(uplifts))); ax.set_yticklabels([f'+{u * 100:.0f}%' for u in uplifts])
        ax.set_title(f'{store} · 발자취 시리즈 (월 {s["monthly_sales"]:.0f}권, 권당 이익 약 {s["margin"]:,.0f}원)\n본전 판매 증가율: 월 10만 원이면 +{s["uplift_needed"] * 100:.0f}%',
                     loc='left', fontsize=12)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel('광고로 늘어난 판매')
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02); cb.set_label('월 손익 (만 원)', labelpad=12); cb.outline.set_visible(False)
    fig.suptitle('광고 손익분기: 시리즈로 묶어도 판매가 20~30% 늘어야 월 10만 원 광고가 본전', x=0.06, ha='left', fontsize=15, y=1.03)
    fig.text(0.06, -0.05, '월 손익 = 월 판매 × 증가율 × 권당 공헌이익 - 광고비. 권당 이익 = 정가×공급율 - 제작원가(20%)·인세(10%)·물류(300원) 가정. 광고 효과 자체는 추정하지 않음.',
             fontsize=10, color=MUTED)
    save(fig, '09_광고_손익분기.png')


if __name__ == '__main__':
    for f in (fig_performance, fig_system, fig_periods, fig_selection, fig_cleaning, fig_errors, fig_agents, fig_readers, fig_breakeven):
        f()
