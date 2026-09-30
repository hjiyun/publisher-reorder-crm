"""교보 자료로 하는 ROI·CRM 분석.

ROI 1  반품 위험 도서 대응의 절감액: 다음 30일 반품 예측 → 매월 상위 20% 대응 → 가정별 순절감액
ROI 2  광고 손익분기: 도서별 권당 공헌이익과 평소 판매량으로 '본전이 되는 판매 증가율' 계산
ROI 3  무작위 판촉 실험 설계: 후보 도서, 층화 배정, 검출 가능 효과 크기
CRM 1  독자 페르소나: 교보 구매자 성별·연령·지역 집계
CRM 2  독자층 기반 도서 추천: 독자 구성이 비슷한 도서
CRM 3  독자층 변화: 6개월 구간별 고객성향 파일이 여러 개일 때 추이

모든 비용 값은 가정이며 ASSUMPTIONS에 모았다. 결과는 가정 조합별 범위로 보고한다.
광고 효과 자체는 추정하지 않는다(집행 기록이 없음). 손익분기는 '얼마나 늘어야 본전인가'만 계산한다.
"""
import json
import math
import re
from itertools import product

import numpy as np
import pandas as pd

from common import CONTACT_FRACTION, STORE, make_models, metrics, pct, table

ASSUMPTIONS = dict(
    return_cost_per_copy=[1000, 2000, 3000],    # 반품 1권당 출판사 비용(회수 물류·검수·훼손 손실), 원
    intervention_success=[0.1, 0.2, 0.3],       # 반품 위험 도서에 대응(공급 축소·재배치 요청)했을 때 막는 반품 비율
    intervention_cost_per_book=[0, 3000],       # 도서 1종 대응 비용(영업 연락 등), 원
    production_cost_ratio=[0.15, 0.20, 0.25],   # 정가 대비 제작원가
    royalty_ratio=0.10,                          # 정가 대비 인세
    logistics_per_copy=300,                      # 권당 출고 물류비, 원
    ad_budgets=[50000, 100000, 300000, 500000],  # 월 광고비 시나리오, 원
    uplift_scenarios=[0.1, 0.3, 0.5],           # '이만큼 늘면'을 가정한 허용 광고비 계산용
)
BASE = dict(return_cost_per_copy=2000, intervention_success=0.2, intervention_cost_per_book=3000, production_cost_ratio=0.20, ad_budget=100000, uplift=0.3)
AGE_BANDS = ['10대 이하', '20대', '30대', '40대', '50대 이상']


# ---------------------------------------------------------------- ROI 1: 반품 위험
def return_rule(df):
    """과잉 공급 규칙: 최근 90일 입하가 최근 3개월 판매보다 많이 쌓인 도서."""
    return (df.rcvd_qty_90d - df.sale_3m).to_numpy(float)


def return_risk(snaps, cols, monthly_pick):
    train, val = snaps[snaps.split == 'train'], snaps[snaps.split == 'val']
    test = snaps[snaps.split == 'test'].reset_index(drop=True)
    validation = {'rule': metrics(val.ret_label, return_rule(val))['precision20']}
    scores = {'rule': return_rule(test)}
    fit = pd.concat([train, val])
    for m, model in make_models().items():
        validation[m] = metrics(val.ret_label, model.fit(train[cols], train.ret_label).predict_proba(val[cols])[:, 1])['precision20']
        scores[m] = model.fit(fit[cols], fit.ret_label).predict_proba(test[cols])[:, 1]
    winner = max((m for m in validation if m != 'rule'), key=validation.get)
    y, q = test.ret_label.to_numpy(), test.ret_qty.to_numpy(float)
    months = test.cutoff.nunique()

    def summary(name, score_list):
        picks = [monthly_pick(test, s, CONTACT_FRACTION) for s in score_list]
        return dict(strategy=name, selected=int(picks[0].sum()), hit_rate=float(np.mean([y[p].mean() for p in picks])),
                    captured_qty=float(np.mean([q[p].sum() for p in picks])))

    label = {'rule': '과잉 공급 규칙', 'logistic': '로지스틱 회귀', 'boosting': '그래디언트 부스팅'}
    expected = scores[winner] * np.maximum(test.rcvd_qty_90d.to_numpy(float), 1)  # 반품은 쌓인 재고를 넘을 수 없다
    strategies = [summary(label[winner] + ' ★ (확률)', [scores[winner]]), summary(label[winner] + ' × 최근 입하량 (기대 반품량)', [expected]),
                  summary(label['rule'], [scores['rule']]),
                  summary('무작위 (30회 평균)', [np.random.default_rng(k).random(len(test)) for k in range(30)])]
    total_qty = float(q.sum())
    for s in strategies:
        s['captured_share'] = s['captured_qty'] / total_qty if total_qty else None
        grid = [(s['captured_qty'] * succ * cost - s['selected'] * icost) / months
                for cost, succ, icost in product(ASSUMPTIONS['return_cost_per_copy'], ASSUMPTIONS['intervention_success'], ASSUMPTIONS['intervention_cost_per_book'])]
        s['saving_base'] = (s['captured_qty'] * BASE['intervention_success'] * BASE['return_cost_per_copy'] - s['selected'] * BASE['intervention_cost_per_book']) / months
        s['saving_min'], s['saving_max'] = min(grid), max(grid)
    return dict(months=months, n=len(test), base_rate=float(y.mean()), total_return_qty=total_qty,
                total_return_cost_base=total_qty * BASE['return_cost_per_copy'] / months, winner=winner,
                validation=validation, strategies=strategies)


# ---------------------------------------------------------------- ROI 2: 광고 손익분기
def breakeven(books, rcvd, sales, months=6):
    last = sales.month.max()
    monthly = sales[sales.month > last - months].groupby('isbn').total.sum() / months
    rate = (rcvd.qty * rcvd.rate).groupby(rcvd.isbn).sum() / rcvd.groupby('isbn').qty.sum()
    b = books.set_index('isbn')
    df = pd.DataFrame({'title': b.title, 'genre': b.genre, 'price': b.price, 'supply_rate': rate, 'monthly_sales': monthly})
    df = df.dropna(subset=['price', 'supply_rate'])
    df['monthly_sales'] = df.monthly_sales.fillna(0)
    a = ASSUMPTIONS
    margin = lambda p: df.price * df.supply_rate / 100 - df.price * (p + a['royalty_ratio']) - a['logistics_per_copy']
    df['margin'] = margin(BASE['production_cost_ratio'])
    df['margin_low'], df['margin_high'] = margin(max(a['production_cost_ratio'])), margin(min(a['production_cost_ratio']))
    for budget in a['ad_budgets']:
        extra = budget / df.margin.where(df.margin > 0)
        df[f'extra_{budget}'] = extra
        df[f'uplift_{budget}'] = (extra / df.monthly_sales).where(df.monthly_sales > 0)
    for u in a['uplift_scenarios']:
        df[f'allow_{u}'] = (u * df.monthly_sales * df.margin).clip(lower=0)  # 판매가 u만큼 늘 때 본전인 월 광고비
    df['series'] = df.title.astype(str).str.replace(r'^(NEW|New|new)\s+', '', regex=True).str.split().str[0]
    counts = []
    for budget in a['ad_budgets']:
        u = df[f'uplift_{budget}']
        counts.append(dict(budget=budget, books_with_sales=int(u.notna().sum()), within_30=int((u <= .3).sum()),
                           within_50=int((u <= .5).sum()), within_100=int((u <= 1).sum())))
    key = f"uplift_{BASE['ad_budget']}"
    au = f"allow_{BASE['uplift']}"
    ser = df.groupby('series').agg(titles=('title', 'size'), monthly_sales=('monthly_sales', 'sum'), margin=('margin', 'mean'), allow=(au, 'sum'))
    ser = ser[ser.titles >= 2].sort_values('allow', ascending=False).head(8)
    series_rows = [dict(series=s, titles=int(r.titles), monthly_sales=r.monthly_sales, margin=r.margin, allow=r.allow,
                        uplift_needed=BASE['ad_budget'] / (r.monthly_sales * r.margin) if r.monthly_sales * r.margin > 0 else None) for s, r in ser.iterrows()]
    top = df[df.monthly_sales >= 3].sort_values(key).head(10)
    top_rows = [dict(title=str(r['title'])[:28], genre=r['genre'], monthly_sales=r['monthly_sales'], supply_rate=r['supply_rate'], margin=r['margin'],
                     extra=r[f"extra_{BASE['ad_budget']}"], uplift=r[key], allow=r[au]) for _, r in top.iterrows()]
    allow_total = {str(u): float(df[f'allow_{u}'].sum()) for u in a['uplift_scenarios']}
    return df.reset_index(names='isbn'), dict(period_months=months, last_month=str(last), counts=counts, top=top_rows, series=series_rows, allow_total=allow_total,
                                              allow_median=float(df.loc[df.monthly_sales >= 3, au].median()),
                                              margin_median=float(df.margin.median()), margin_range=[float(df.margin_low.median()), float(df.margin_high.median())])


# ---------------------------------------------------------------- ROI 3: 실험 설계
def experiment(be, sales, n=20, seed=2026):
    key = f"uplift_{BASE['ad_budget']}"
    cand = be[(be.monthly_sales >= 3) & be[key].notna()].sort_values(key).head(n).copy()
    if len(cand) < 6:
        return None
    rng = np.random.default_rng(seed)
    cand['stratum'] = cand.genre + '/' + np.where(cand.monthly_sales >= cand.monthly_sales.median(), '판매 상', '판매 하')
    cand['group'] = ''
    for _, idx in cand.groupby('stratum').groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            cand.loc[j, 'group'] = '판촉' if i % 2 == 0 else '비교'
    # 검출 가능 효과: 후보 도서의 월 판매 로그 변화(전월 대비) 표준편차로 근사
    piv = sales[sales.isbn.isin(cand.isbn)].pivot_table(index='isbn', columns='month', values='total', aggfunc='sum').fillna(0)
    piv = piv.iloc[:, -25:].clip(lower=0)  # 반품 상계로 음수인 달은 0으로 본다
    sd = float(np.log1p(piv).diff(axis=1).iloc[:, 1:].stack().std())
    k = min((cand.group == '판촉').sum(), (cand.group == '비교').sum())
    mde = lambda post_months: math.exp(2.8 * sd / math.sqrt(post_months) * math.sqrt(2 / k)) - 1
    rows = [dict(title=str(r.title)[:28], genre=r.genre, monthly_sales=r.monthly_sales, stratum=r.stratum, group=r.group) for r in cand.itertuples()]
    return dict(n=len(cand), per_arm=int(k), sd_log_change=sd, mde_1m=mde(1), mde_2m=mde(2), seed=seed, rows=rows)


# ---------------------------------------------------------------- CRM
def age_band(label):
    m = re.match(r'(\d+)', str(label))
    if not m:
        return '기타'
    a = int(m.group(1))
    return '10대 이하' if a < 20 else '20대' if a < 30 else '30대' if a < 40 else '40대' if a < 50 else '50대 이상'


def reader_frame(src):
    """고객성향 파일(6개월 구간별)을 도서×구간 표로 편다."""
    rows = []
    for path in sorted((src.parent / '고객성향').glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf8'))
        period = '~'.join(payload.get('period', [path.stem, '']))
        for r in payload['rows']:
            g = r.get('gender', {})
            row = dict(period=period, isbn=r['isbn'], male=g.get('남자', 0), female=g.get('여자', 0), unknown=g.get('기타', 0))
            for band in AGE_BANDS + ['기타']:
                row[f'age_{band}'] = 0
            bulk = (None, 0)
            for k, v in r.get('age_total', {}).items():
                row[f'age_{age_band(k)}'] += v
                if re.match(r'\d', str(k)) and v > bulk[1]:
                    bulk = (k, v)
            known_age_raw = sum(v for k, v in r.get('age_total', {}).items() if re.match(r'\d', str(k)))
            # 한 연령 구간에 100권 이상이 몰리면 기관·대량 구매로 본다(개인 구매 분포로 해석하지 않음)
            row['bulk_label'], row['bulk_copies'] = (bulk[0], bulk[1]) if bulk[1] >= 100 and bulk[1] >= .5 * known_age_raw else ('', 0)
            for k, v in r.get('region', {}).items():
                row[f'region_{k}'] = v
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows).fillna(0)
    df['copies'] = df.male + df.female + df.unknown
    df['known'] = df.male + df.female
    df['known_age'] = df[[f'age_{b}' for b in AGE_BANDS]].sum(axis=1)
    top = df[[f'age_{b}' for b in AGE_BANDS]].idxmax(axis=1).str[4:]
    fshare = (df.female / df.known).where(df.known > 0)
    df['persona'] = np.where((df.known < 5) | (df.known_age == 0), '표본 부족',
                             top + ' ' + np.where(fshare >= .6, '여성', np.where(fshare <= .4, '남성', '남녀')) + ' 중심')
    return df


def personas(df, books):
    latest = df[df.period == df.period.max()].merge(books[['isbn', 'genre', 'title']], on='isbn', how='left')
    region_cols = [c for c in latest.columns if c.startswith('region_')]
    regions = latest[region_cols].sum().sort_values(ascending=False)
    total_region = regions.sum()
    region_rows = [dict(region=c[7:], copies=int(v), share=v / total_region) for c, v in regions.head(8).items()] if total_region else []
    genre_rows = []
    for g, part in latest.groupby('genre'):
        known_age = part[[f'age_{b}' for b in AGE_BANDS]].sum()
        genre_rows.append(dict(genre=g, titles=len(part), copies=int(part.copies.sum()), known_share=part.known.sum() / part.copies.sum(),
                               female_share=part.female.sum() / part.known.sum() if part.known.sum() else None,
                               top_age=known_age.idxmax()[4:] if known_age.sum() else '-',
                               top_age_share=known_age.max() / known_age.sum() if known_age.sum() else None))
    genre_rows.sort(key=lambda r: -r['copies'])
    known = latest.known.sum()
    age_known = latest[[f'age_{b}' for b in AGE_BANDS]].sum()
    return dict(period=latest.period.iloc[0], titles=len(latest), copies=int(latest.copies.sum()), known_share=known / latest.copies.sum(),
                female_share=latest.female.sum() / known if known else None,
                age_share={b: float(age_known[f'age_{b}'] / age_known.sum()) for b in AGE_BANDS} if age_known.sum() else {},
                regions=region_rows, genres=genre_rows)


def similar_books(df, books, k=3):
    latest = df[(df.period == df.period.max()) & (df.known >= 5)].reset_index(drop=True)
    if len(latest) < k + 1:
        return []
    region_cols = [c for c in latest.columns if c.startswith('region_')]
    top_regions = latest[region_cols].sum().sort_values(ascending=False).index[:6]
    ages = latest[[f'age_{b}' for b in AGE_BANDS]].div(latest.known_age.where(latest.known_age > 0), axis=0).fillna(0)
    reg = latest[list(top_regions)].div(latest[region_cols].sum(axis=1).where(lambda s: s > 0), axis=0).fillna(0)
    vec = np.column_stack([(latest.female / latest.known).to_numpy(), ages.to_numpy(), reg.to_numpy() * .5])
    vec = (vec - vec.mean(axis=0)) / vec.std(axis=0).clip(1e-9)  # 독자 구성이 서로 비슷하면 원값 그대로는 유사도가 한쪽으로 몰린다
    unit = vec / np.linalg.norm(vec, axis=1, keepdims=True).clip(1e-9)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -1)
    title = books.set_index('isbn').title
    out = []
    for i, row in latest.iterrows():
        best = np.argsort(-sim[i])[:k]
        out.append(dict(title=str(title.get(row.isbn, row.isbn))[:26], persona=row.persona, copies=int(row.copies),
                        similar=' / '.join(f'{str(title.get(latest.isbn[j], latest.isbn[j]))[:18]} ({sim[i, j]:.2f})' for j in best)))
    out.sort(key=lambda r: -r['copies'])
    return out[:15]


def reader_trend(df):
    rows = []
    for period, part in df.groupby('period'):
        known, age_known = part.known.sum(), part[[f'age_{b}' for b in AGE_BANDS]].sum()
        top_persona = part.groupby('persona').copies.sum().drop('표본 부족', errors='ignore')
        rows.append(dict(period=period, titles=len(part), copies=int(part.copies.sum()), known_share=known / part.copies.sum() if part.copies.sum() else None,
                         female_share=part.female.sum() / known if known else None,
                         **{f'age_{b}': float(age_known[f'age_{b}'] / age_known.sum()) if age_known.sum() else None for b in AGE_BANDS},
                         top_persona=top_persona.idxmax() if len(top_persona) else '-',
                         top_persona_share=float(top_persona.max() / part.copies.sum()) if len(top_persona) else None,
                         note='성별·연령 전부 미기록 (원자료 결측)' if known == 0 else
                              '; '.join(f'{b} 구간 {int(c)}권 대량 구매 의심' for b, c in part.loc[part.bulk_copies > 0, ['bulk_label', 'bulk_copies']].itertuples(index=False)) or ''))
    return rows


def analyze(src, books, rcvd, sales, snaps, cols, monthly_pick):
    be, be_summary = breakeven(books, rcvd, sales)
    out = dict(assumptions=ASSUMPTIONS, base=BASE, returns=return_risk(snaps, cols, monthly_pick), breakeven=be_summary,
               experiment=experiment(be, sales))
    df = reader_frame(src)
    if df is not None:
        out.update(personas=personas(df, books), similar=similar_books(df, books), trend=reader_trend(df))
    return out


# ---------------------------------------------------------------- 보고서 조각
won = lambda v: '–' if v is None else f'{v:,.0f}원'


def sections(r):
    a, b, rr, be = r['assumptions'], r['base'], r['returns'], r['breakeven']
    html = []
    rc = [('strategy', '전략', None), ('selected', '대응 도서×월', lambda v: f'{v:,}'), ('hit_rate', '실제 반품 비율', pct),
          ('captured_qty', '포착한 반품(권)', lambda v: f'{v:,.0f}'), ('captured_share', '전체 반품 중', pct),
          ('saving_base', '월 순절감 (기준 가정)', won), ('saving_min', '최소', won), ('saving_max', '최대', won)]
    html.append(f'''<section><h2>ROI 1 · 반품 위험 도서 대응의 절감액</h2>
<p class="note">매월 1일, 다음 30일 안에 {STORE["name"]} 반품이 생길 도서를 예측하고 상위 20%에 대응(공급 축소·재배치 요청 등)한다고 가정한다. 평가 {rr["months"]}개월 동안 반품 {rr["total_return_qty"]:,.0f}권, 도서×월 {rr["n"]:,}건 중 반품 발생 {pct(rr["base_rate"])}.
기준 가정: 반품 1권당 비용 {b["return_cost_per_copy"]:,}원, 대응 시 {pct(b["intervention_success"], 0)} 방지, 대응 비용 도서당 {b["intervention_cost_per_book"]:,}원 → 대응 없을 때 월 반품 비용 {won(rr["total_return_cost_base"])}.
최소·최대는 비용 {a["return_cost_per_copy"]}원 × 방지율 {a["intervention_success"]} × 대응 비용 {a["intervention_cost_per_book"]}원 조합의 범위다. ★ 검증 구간에서 선택된 모델.</p>
{table(rr["strategies"], rc)}</section>''')

    cc = [('budget', '월 광고비', won), ('books_with_sales', '판매 있는 도서', lambda v: f'{v:,}'), ('within_30', '30% 이하 증가로 본전', lambda v: f'{v:,}종'),
          ('within_50', '50% 이하', lambda v: f'{v:,}종'), ('within_100', '2배 이하', lambda v: f'{v:,}종')]
    tc = [('title', '도서', None), ('genre', '분야', None), ('monthly_sales', '평소 월 판매', lambda v: f'{v:,.1f}권'), ('supply_rate', '공급율', lambda v: f'{v:.0f}%'),
          ('margin', '권당 공헌이익', won), ('extra', '본전 추가 판매', lambda v: f'{v:,.0f}권'), ('uplift', '필요 증가율', pct), ('allow', f'30% 증가 시 허용 광고비', won)]
    sc2 = [('series', '시리즈(도서명 첫 단어)', None), ('titles', '도서', lambda v: f'{v}종'), ('monthly_sales', '월 판매 합', lambda v: f'{v:,.1f}권'),
           ('margin', '평균 공헌이익', won), ('allow', '30% 증가 시 허용 광고비', won), ('uplift_needed', f'월 {b["ad_budget"]:,}원 본전 증가율', pct)]
    html.append(f'''<section><h2>ROI 2 · 광고 손익분기</h2>
<p class="note">권당 공헌이익 = 정가×공급율 − 정가×(제작원가 {pct(b["production_cost_ratio"], 0)} + 인세 {pct(a["royalty_ratio"], 0)}) − 물류 {a["logistics_per_copy"]}원. 중앙값 {won(be["margin_median"])} (제작원가 가정 {a["production_cost_ratio"]} 범위: {won(be["margin_range"][0])} ~ {won(be["margin_range"][1])}).
평소 판매 = 최근 {be["period_months"]}개월 {STORE["name"]} 월평균. {STORE["name"]} 외 판매는 포함하지 않으므로 필요 증가율은 실제보다 크게 나온다(보수적). 광고 효과를 추정한 것이 아니라 본전 조건만 계산했다.</p>
<p><b>허용 광고비</b>: 판매가 30% 는다고 가정하면 도서 1종에 쓸 수 있는 월 광고비는 중앙값 {won(be["allow_median"])}, 전 도서 합계 {won(be["allow_total"]["0.3"])} (10%: {won(be["allow_total"]["0.1"])}, 50%: {won(be["allow_total"]["0.5"])}).</p>
{table(be["counts"], cc)}<h3>월 {b["ad_budget"]:,}원 광고 시 본전이 가장 쉬운 도서 (평소 월 3권 이상)</h3>{table(be["top"], tc)}
<h3>시리즈 단위로 묶으면</h3><p class="note">광고를 시리즈 전체에 걸면 판매 기반이 커져 본전 조건이 낮아진다. 시리즈는 도서명 첫 단어로 근사했다(2종 이상).</p>{table(be["series"], sc2)}</section>''')

    ex = r.get('experiment')
    if ex:
        ec = [('title', '도서', None), ('genre', '분야', None), ('monthly_sales', '평소 월 판매', lambda v: f'{v:,.1f}권'), ('stratum', '층', None), ('group', '배정', None)]
        html.append(f'''<section><h2>ROI 3 · 무작위 판촉 실험 설계 (출판사 협조 시)</h2>
<p class="note">후보: ROI 2에서 본전이 쉬운 도서 {ex["n"]}종. 분야×판매 규모로 층을 나눈 뒤 층마다 절반을 판촉에 무작위 배정했다(시드 {ex["seed"]}, 재현 가능).
판촉 기간 동안 두 그룹의 {STORE["name"]} 판매 변화를 비교한다. 과거 월 판매 로그 변화의 표준편차 {ex["sd_log_change"]:.2f} 기준, 그룹당 {ex["per_arm"]}종이면
1개월 비교로 약 {pct(ex["mde_1m"], 0)}, 2개월 평균으로 약 {pct(ex["mde_2m"], 0)} 이상의 판매 증가를 검출할 수 있다(검정력 80%, 유의수준 5% 근사).
그보다 작은 효과는 '효과 없음'과 구별하기 어렵다.</p>{table(ex["rows"], ec)}</section>''')

    pe = r.get('personas')
    if pe:
        ages = ' · '.join(f'{k} {pct(v, 0)}' for k, v in pe['age_share'].items())
        gc = [('genre', '분야', None), ('titles', '도서', lambda v: f'{v:,}종'), ('copies', '판매(권)', lambda v: f'{v:,}'), ('known_share', '성별 확인', pct),
              ('female_share', '여성 비율', pct), ('top_age', '주 연령', None), ('top_age_share', '주 연령 비중', pct)]
        rgc = [('region', '지역', None), ('copies', '권', lambda v: f'{v:,}'), ('share', '비중', pct)]
        html.append(f'''<section><h2>CRM 1 · 독자 페르소나 ({pe["period"]})</h2>
<p class="note">{STORE["name"]} 구매자 집계. 도서 {pe["titles"]}종, {pe["copies"]:,}권. 성별 확인 {pct(pe["known_share"])}, 확인분 중 여성 {pct(pe["female_share"])}. 연령(확인분): {ages}.
구매자 연령·성별과 실제 사용자는 다를 수 있다. 개인 단위가 아닌 집계 결과다.</p>
<h3>분야별</h3>{table(pe["genres"], gc)}<h3>지역 (상위 8)</h3>{table(pe["regions"], rgc)}</section>''')
    if r.get('similar'):
        sc = [('title', '도서', None), ('persona', '독자층', None), ('copies', '판매(권)', lambda v: f'{v:,}'), ('similar', '독자 구성이 비슷한 도서 (유사도)', None)]
        html.append(f'''<section><h2>CRM 2 · 독자층 기반 도서 추천</h2>
<p class="note">성별·연령·지역 구성을 도서 간 표준화한 뒤 코사인 유사도(−1~1). 성별 확인 5권 이상 도서만. 교차판매·묶음 기획 후보이며 개인 구매 이력 기반 추천이 아니다. 판매 상위 15종.</p>{table(r["similar"], sc)}</section>''')
    if r.get('trend'):
        tr = [('period', '구간', None), ('titles', '도서', lambda v: f'{v:,}'), ('copies', '판매(권)', lambda v: f'{v:,}'), ('known_share', '성별 확인', pct),
              ('female_share', '여성', pct), *[(f'age_{b}', b, pct) for b in AGE_BANDS], ('top_persona', '최대 독자층', None), ('top_persona_share', '비중', pct), ('note', '품질 메모', None)]
        note = '구간이 1개뿐이라 추이는 아직 볼 수 없다. 6개월 구간별 고객성향을 더 받으면 채워진다.' if len(r['trend']) < 2 else f'6개월 구간별 {STORE["name"]} 구매자 구성. 연령은 확인분 기준. 최대 독자층 비중은 그 독자층으로 분류된 도서의 판매 비중이다. 품질 메모의 구간은 해석에서 제외하거나 따로 본다.'
        html.append(f'<section><h2>CRM 3 · 독자층 변화</h2><p class="note">{note}</p>{table(r["trend"], tr)}</section>')
    return '\n'.join(html)
