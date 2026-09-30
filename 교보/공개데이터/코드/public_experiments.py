"""공개 데이터 확장 실험: "출판사가 이런 데이터를 모은다면 무엇을 할 수 있나".

실험 A  Hillstrom 이메일 캠페인(무작위 배정) → 판촉 효과(ATE), 업리프트 타깃팅, 메시지 ROI 시나리오
실험 B  Book-Crossing 도서 평점(사용자 연령 포함) → 집계 페르소나 추천 vs 개인 이력 기반 추천 비교

실행:  python public_experiments.py      결과: ../결과/<run_id>/index.html
회사 자료와 섞지 않는다. 출판사 수치는 교보 분석 결과(권당 공헌이익 중앙값)만 가정값으로 가져온다.
"""
import glob
import html
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
KYOBO_RESULTS = ROOT.parent.parent / '결과'  # 교보/결과 (권당 공헌이익 가정값을 가져온다)
SEED = 2026
FRACTION = 0.2
MESSAGE_COST = [10, 20, 50]  # 메시지 1건 발송 비용 가정(원). 카카오 알림톡·문자 수준
AGE_BANDS = [(0, 19, '10대 이하'), (20, 29, '20대'), (30, 39, '30대'), (40, 49, '40대'), (50, 200, '50대 이상')]


def ci_diff(a, b):
    """두 그룹 평균 차이와 95% 정규 근사 구간."""
    d = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(d), float(d - 1.96 * se), float(d + 1.96 * se)


# ---------------------------------------------------------------- 실험 A: Hillstrom
def hillstrom(margin):
    df = pd.read_csv(DATA / 'hillstrom' / 'hillstrom.csv')
    control = df[df.segment == 'No E-Mail']
    ate = []
    for seg in ('Mens E-Mail', 'Womens E-Mail'):
        t = df[df.segment == seg]
        for y in ('visit', 'conversion', 'spend'):
            d, lo, hi = ci_diff(t[y], control[y])
            ate.append(dict(segment=seg, outcome=y, control=float(control[y].mean()), treated=float(t[y].mean()), diff=d, lo=lo, hi=hi,
                            relative=d / control[y].mean() if control[y].mean() else None))
    # 업리프트: 메일 발송(남·여 합침) vs 미발송, T-learner
    df = df.assign(treat=(df.segment != 'No E-Mail').astype(int))
    X = pd.get_dummies(df[['recency', 'history', 'mens', 'womens', 'newbie', 'zip_code', 'channel']], columns=['zip_code', 'channel'], dtype=float)
    tr, te = train_test_split(df.index, test_size=0.3, random_state=SEED, stratify=df.treat)
    models = {}
    for arm in (1, 0):
        idx = [i for i in tr if df.treat[i] == arm]
        models[arm] = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED).fit(X.loc[idx], df.visit[idx])
    test = df.loc[te].copy()
    test['uplift'] = models[1].predict_proba(X.loc[te])[:, 1] - models[0].predict_proba(X.loc[te])[:, 1]
    rank = test.sort_values('uplift', ascending=False)
    curve = []
    for f in (0.1, 0.2, 0.3, 0.5, 1.0):
        top = rank.head(max(1, int(len(rank) * f)))
        row = dict(fraction=f, n=len(top))
        for y in ('visit', 'conversion', 'spend'):
            t, c = top[top.treat == 1][y], top[top.treat == 0][y]
            row[y] = float(t.mean() - c.mean()) if len(t) and len(c) else None
        curve.append(row)
    top20 = next(r for r in curve if r['fraction'] == FRACTION)
    everyone = next(r for r in curve if r['fraction'] == 1.0)
    # ROI 시나리오: 전환 1건 = 도서 1권 구매로 가정, 권당 공헌이익은 교보 분석 중앙값
    roi = []
    for label, uplift_conv in (('전체 발송', everyone['conversion']), (f'업리프트 상위 {int(FRACTION * 100)}%만 발송', top20['conversion'])):
        for cost in MESSAGE_COST:
            per1000 = 1000 * uplift_conv * margin - 1000 * cost
            roi.append(dict(target=label, cost=cost, uplift_conv=uplift_conv, profit_per_1000=per1000,
                            roi=per1000 / (1000 * cost) if cost else None, breakeven_cost=uplift_conv * margin))
    return dict(n=len(df), control_rate=dict(visit=float(control.visit.mean()), conversion=float(control.conversion.mean())), ate=ate,
                curve=curve, roi=roi, margin=margin, test_n=len(test))


# ---------------------------------------------------------------- 실험 B: Book-Crossing
def age_band(a):
    for lo, hi, name in AGE_BANDS:
        if lo <= a <= hi:
            return name
    return None


def bookcrossing(min_book=20, min_user=5, k=10):
    path = DATA / 'bookcrossing'
    ratings = pd.read_csv(path / 'BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv(path / 'BX-Users.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
    books = pd.read_csv(path / 'BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip', usecols=[0, 1, 2], dtype=str)
    ratings.columns, users.columns = ['user', 'isbn', 'rating'], ['user', 'location', 'age']
    raw = dict(ratings=len(ratings), users=len(users), explicit=int((ratings.rating > 0).sum()))
    r = ratings[ratings.rating > 0]
    users = users[users.age.between(5, 100)].assign(band=lambda d: d.age.map(age_band))
    r = r[r.user.isin(users.user)]
    for _ in range(3):  # 도서·사용자 최소 건수 필터를 몇 번 반복해 안정화
        r = r[r.isbn.map(r.isbn.value_counts()) >= min_book]
        r = r[r.user.map(r.user.value_counts()) >= min_user]
    band = users.set_index('user').band
    persona = r.assign(band=r.user.map(band)).groupby('band').agg(ratings=('rating', 'size'), users=('user', 'nunique'), mean_rating=('rating', 'mean'))
    persona['share'] = persona.ratings / persona.ratings.sum()
    # 사용자별 20% 보류(최소 1건) → 보류한 책을 맞히는지
    rng = np.random.default_rng(SEED)
    r = r.sample(frac=1, random_state=SEED).reset_index(drop=True)
    r['pos'] = r.groupby('user').cumcount()
    r['n'] = r.user.map(r.user.value_counts())
    test_mask = r.pos < np.maximum(1, (r.n * 0.2).astype(int))
    train, test = r[~test_mask], r[test_mask]
    uid = {u: i for i, u in enumerate(sorted(r.user.unique()))}
    iid = {b: i for i, b in enumerate(sorted(r.isbn.unique()))}
    X = sparse.csr_matrix((np.ones(len(train)), (train.user.map(uid), train.isbn.map(iid))), shape=(len(uid), len(iid)))
    popularity = np.asarray(X.sum(axis=0)).ravel()
    ubands = pd.Series(r.user.map(band).values, index=r.user.values).groupby(level=0).first()
    band_pop = {}
    for b in ubands.dropna().unique():
        rows = [uid[u] for u in ubands[ubands == b].index]
        band_pop[b] = np.asarray(X[rows].sum(axis=0)).ravel()
    # 아이템 기반 협업 필터링(코사인)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel()).clip(1e-9)
    S = (X.T @ X).astype(float)
    S = sparse.diags(1 / norms) @ S @ sparse.diags(1 / norms)
    S.setdiag(0)
    S = S.tocsr()
    truth = test.groupby('user').isbn.apply(lambda s: set(s.map(iid)))
    results = {m: dict(hit=0, prec=0.0, rec=0.0, items=set()) for m in ('전체 인기순', '연령대(페르소나) 인기순', '개인 이력 기반 협업 필터링')}
    for u, items in truth.items():
        row = uid[u]
        seen = X[row].indices
        scores = {'전체 인기순': popularity.astype(float), '연령대(페르소나) 인기순': band_pop.get(ubands.get(u), popularity).astype(float),
                  '개인 이력 기반 협업 필터링': np.asarray((X[row] @ S).todense()).ravel() + popularity * 1e-9}
        for m, s in scores.items():
            s = s.copy()
            s[seen] = -np.inf
            top = np.argpartition(-s, k)[:k]
            hits = len(items & set(top))
            res = results[m]
            res['hit'] += hits > 0
            res['prec'] += hits / k
            res['rec'] += hits / len(items)
            res['items'].update(top.tolist())
    n = len(truth)
    table = [dict(method=m, hit_rate=v['hit'] / n, precision=v['prec'] / n, recall=v['rec'] / n, coverage=len(v['items']) / len(iid)) for m, v in results.items()]
    title = books.set_index(books.columns[0])[books.columns[1]]
    top_by_band = []
    for b in [x[2] for x in AGE_BANDS]:
        if b in band_pop:
            inv = {i: s for s, i in iid.items()}
            best = np.argsort(-band_pop[b])[:3]
            top_by_band.append(dict(band=b, books=' / '.join(html.unescape(str(title.get(inv[i], inv[i])))[:30] for i in best)))
    return dict(raw=raw, used=dict(ratings=len(r), users=len(uid), books=len(iid), test_users=n), k=k, min_book=min_book, min_user=min_user,
                persona=persona.reset_index().to_dict('records'), results=table, top_by_band=top_by_band)


# ---------------------------------------------------------------- 보고서
CSS = '''body{font-family:"Malgun Gothic",Arial,sans-serif;background:#f4f5f1;color:#152b32;max-width:1080px;margin:40px auto;padding:0 20px;line-height:1.7}
h1{font-size:30px;line-height:1.4}section{background:#fff;padding:24px 28px;border-radius:12px;margin:20px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 10px;border-bottom:1px solid #e3e7e1;text-align:left}th{background:#17354b;color:#fff;white-space:nowrap}
td.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}.note{color:#5b6b70;font-size:14px}.banner{background:#fff4d6}'''
pct = lambda v, d=1: '–' if v is None else f'{v * 100:.{d}f}%'
pp = lambda v: '–' if v is None else f'{v * 100:+.2f}%p'
won = lambda v: '–' if v is None else f'{v:,.0f}원'


def table(rows, cols):
    head = ''.join(f'<th>{html.escape(l)}</th>' for _, l, _ in cols)
    body = ''.join('<tr>' + ''.join(f'<td class="{"num" if f else ""}">{html.escape(f(r.get(k)) if f else str(r.get(k, "")))}</td>' for k, _, f in cols) + '</tr>' for r in rows)
    return f'<table><tr>{head}</tr>{body}</table>'


def build_report(out, a, b):
    ate_c = [('segment', '처치', None), ('outcome', '결과', None), ('control', '미발송', lambda v: f'{v:.4f}'), ('treated', '발송', lambda v: f'{v:.4f}'),
             ('diff', '차이', lambda v: f'{v:+.4f}'), ('lo', '95% 하한', lambda v: f'{v:+.4f}'), ('hi', '95% 상한', lambda v: f'{v:+.4f}'), ('relative', '상대 변화', pct)]
    cur_c = [('fraction', '발송 대상 (업리프트 상위)', lambda v: pct(v, 0)), ('n', '고객 수', lambda v: f'{v:,}'), ('visit', '방문 증가', pp),
             ('conversion', '구매 전환 증가', pp), ('spend', '구매액 증가($)', lambda v: '–' if v is None else f'{v:+.3f}')]
    roi_c = [('target', '발송 방식', None), ('cost', '건당 비용', won), ('uplift_conv', '전환 증가', pp), ('profit_per_1000', '1,000건당 순이익', won),
             ('roi', 'ROI', pct), ('breakeven_cost', '본전 건당 비용', won)]
    per_c = [('band', '연령대', None), ('users', '사용자', lambda v: f'{v:,}'), ('ratings', '평점', lambda v: f'{v:,}'), ('share', '비중', pct), ('mean_rating', '평균 평점', lambda v: f'{v:.2f}')]
    res_c = [('method', '추천 방식', None), ('hit_rate', f'적중률@{b["k"]}', pct), ('precision', f'정밀도@{b["k"]}', lambda v: pct(v, 2)),
             ('recall', f'재현율@{b["k"]}', pct), ('coverage', '추천된 도서 다양성', pct)]
    body = f'''<p class="note">공개 데이터 확장 실험 · 실행 {out.name}</p>
<h1>공개 데이터 실험: 출판사가 이런 데이터를 모은다면</h1>
<section class="banner"><b>회사 자료가 아닌 공개 데이터 실험이다.</b> 다른 나라·업종·시기의 데이터이므로 효과 크기가 출판사에 그대로 적용된다는 보장은 없다.
출판사 수치는 교보 분석의 권당 공헌이익 중앙값({won(a["margin"])})만 가정값으로 가져와 시나리오를 만들었다.</section>
<section><h2>실험 A · 판촉 메시지의 효과 (Hillstrom, 무작위 배정 {a["n"]:,}명)</h2>
<p class="note">미국 소매업체가 고객을 무작위로 세 그룹(남성용 메일·여성용 메일·미발송)으로 나눠 2주간 결과를 본 실험. 무작위 배정이라 차이를 메일의 효과로 해석할 수 있다.</p>
{table(a["ate"], ate_c)}
<h3>누구에게 보내야 하나 (업리프트 모델)</h3><p class="note">메일 발송/미발송 각각으로 방문 확률 모델을 학습해 그 차이(업리프트)로 고객을 줄 세웠다. 평가용 {a["test_n"]:,}명에서 상위 고객일수록 효과가 큰지 확인한다.</p>
{table(a["curve"], cur_c)}
<h3>출판사에 적용하면: 메시지 ROI 시나리오</h3><p class="note">출판사가 구매자 연락처(카카오 채널·이메일)를 모은다고 가정한다. 전환 1건 = 도서 1권, 권당 공헌이익 {won(a["margin"])}.
건당 비용은 {MESSAGE_COST}원을 가정했다. '본전 건당 비용'보다 싸게 보낼 수 있으면 이익이다.</p>{table(a["roi"], roi_c)}</section>
<section><h2>실험 B · 집계 페르소나 추천 vs 개인 이력 추천 (Book-Crossing)</h2>
<p class="note">도서 평점 공개 데이터. 원본 평점 {b["raw"]["ratings"]:,}건 중 명시 평점·연령 확인 사용자·최소 건수(도서 {b["min_book"]}건, 사용자 {b["min_user"]}건) 조건으로
평점 {b["used"]["ratings"]:,}건, 사용자 {b["used"]["users"]:,}명, 도서 {b["used"]["books"]:,}종을 썼다. 사용자마다 평가한 책의 20%를 가리고 상위 {b["k"]}권 추천에 들어가는지 본다.</p>
<h3>연령대별 독자 구성</h3>{table(b["persona"], per_c)}
<h3>추천 성능</h3><p class="note">'연령대 인기순'은 교보 고객성향처럼 집계 자료만 있을 때 할 수 있는 추천이고, '개인 이력 기반'은 개인 구매 이력이 있어야 가능한 추천이다.</p>{table(b["results"], res_c)}
<h3>연령대별 인기 도서 (상위 3)</h3>{table(b["top_by_band"], [('band', '연령대', None), ('books', '도서', None)])}</section>
<section><h2>해석과 한계</h2><ul>
<li>실험 A: 무작위 실험이므로 인과 효과지만, 2008년 미국 의류·잡화 소매업체의 결과다. 출판사 적용은 전환 증가율이 비슷하다는 가정에 기댄다.</li>
<li>실험 B: 평점은 구매와 다르고 2004년 이전 자료다. 성능 차이는 '개인 이력 데이터가 있으면 이만큼 나아질 수 있다'는 방향을 보여줄 뿐이다.</li>
<li>두 실험 모두 출판사가 직접 고객 연락처·구매 이력을 모아야 실제로 할 수 있다.</li></ul></section>'''
    path = out / 'index.html'
    path.write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공개 데이터 실험</title><style>{CSS}</style>{body}</html>', encoding='utf8')
    return path


def kyobo_margin(default=4600.0):
    files = sorted(glob.glob(str(KYOBO_RESULTS / '*' / '04_crm_roi.json')))
    if not files:
        return default
    return float(json.loads(Path(files[-1]).read_text(encoding='utf8'))['breakeven']['margin_median'])


if __name__ == '__main__':
    t0 = time.time()
    out = ROOT.parent / '결과' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    out.mkdir(parents=True)
    a = hillstrom(kyobo_margin())
    b = bookcrossing()
    (out / 'results.json').write_text(json.dumps(dict(hillstrom=a, bookcrossing=b), ensure_ascii=False, default=str), encoding='utf8')
    print(build_report(out, a, b), f'{time.time() - t0:.0f}s')
