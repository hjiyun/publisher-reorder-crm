"""예스24 SCM 자료에 교보 분석과 같은 절차(본 목표: 다음 30일 재주문 도서 상위 20% 선정)를 적용한다.

실행:  python yes24.py        입력 ../원본/ → 결과 ../결과/<run_id>/index.html

교보와 다른 점은 자료를 읽는 부분뿐이다(분석·평가·ROI·CRM 코드는 ../../교보/코드/를 그대로 쓴다).
- 재주문 = 예스24 발주(일자별발주목록, 파주+대구 발주수량). 교보는 발주 이력이 없어 입하로 대신했다.
- 반품 = 반출(반출확정일, 반출수량, 반출사유).
- 판매 = 월별 상품매출(예스24는 온라인 서점이라 전부 '온라인' 판매로 둔다).
- 도서 기준정보 = 교보 도서정보(출판사 도서 목록, 분야·출간일) + 예스24에만 있는 도서.
- 공급율 = 예스24 입고내역의 공급율(ROI 손익분기 계산용).
- 구매자 = 월별 상품매출의 성별·연령·지역 수치를 6개월 구간 파일로 바꿔 ../고객성향/에 둔다.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / '원본'
KYOBO_CODE = ROOT.parent.parent / '교보' / '코드'
sys.path.insert(0, str(KYOBO_CODE))

import common  # noqa: E402
import kyobo  # noqa: E402

KYOBO_LOAD = kyobo.load  # main()에서 kyobo.load를 바꾸기 전에 원래 함수를 잡아 둔다

AGE = {'UNDER10': '0~9', 'EARLY20': '20~24', 'AFTER20': '25~29', 'EARLY30': '30~34', 'AFTER30': '35~39',
       'EARLY40': '40~44', 'AFTER40': '45~49', 'EARLY50': '50~54', 'AFTER50': '55~59', 'ABOVE60': '60~'}
REGION = {'SEOUL': '서울특별시', 'KYOUNGGI': '경기도', 'CHUNGCHEONG': '충청도', 'KYOUNGSANG': '경상도',
          'JEOLLA': '전라도', 'GANGWON': '강원도', 'JEJU': '제주도'}
num = lambda s: pd.to_numeric(s, errors='coerce')


def rows(path, key='rows'):
    return json.loads(path.read_text(encoding='utf8'))[key]


def load(src):
    files = {p.name: p for p in src.glob('*.json')}
    hashes = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files.items()}
    pick = lambda prefix: next(p for n, p in files.items() if n.startswith(prefix))
    goods = pd.DataFrame(rows(pick('상품목록')))
    goods['isbn'] = goods.EAN2.astype(str).str.strip()
    to_isbn = dict(zip(goods.GOODS_NO.astype(str), goods.isbn))

    # 도서 기준정보: 출판사 도서 목록(교보 도서정보) + 예스24에만 있는 도서
    kb = KYOBO_LOAD(kyobo.SRC)[0]
    extra = goods[~goods.isbn.isin(kb.isbn)]
    pub_col = next((c for c in ('SALE_WILL_DTS', 'REG_DTS') if c in goods.columns), None)
    extra_books = pd.DataFrame({'isbn': extra.isbn, 'title': extra.GOODS_NM,
                                'pub': pd.to_datetime(extra[pub_col], errors='coerce') if pub_col else pd.NaT,
                                'price': num(extra.SHOP_PR), 'genre': '기타', 'status': ''})
    books = pd.concat([kb, extra_books], ignore_index=True)

    # 재주문 = 발주 (파주 + 대구)
    it = pd.DataFrame(json.loads(pick('발주내역').read_text(encoding='utf8'))['items'])
    rate = pd.DataFrame(rows(pick('입고내역')))
    rate = num(rate.GOODS_WH_RT).groupby(rate.EAN2.astype(str)).median()
    orders = pd.DataFrame({'isbn': it.GOODS_NO.astype(str).map(to_isbn), 'date': pd.to_datetime(it.PUR_DTS, errors='coerce'),
                           'qty': num(it.PUR_CNT_001).fillna(0) + num(it.PUR_CNT_003).fillna(0), 'buy': '',
                           'center': it.WH_CODE.astype(str), 'doc': it.PUR_NO_STR.astype(str) + '-' + it.PUR_SEQ.astype(str)})
    orders['rate'] = orders.isbn.map(rate)

    r = pd.DataFrame(rows(pick('반출내역')))
    rtgd = pd.DataFrame({'isbn': r.EAN2.astype(str), 'date': pd.to_datetime(r.EOUT_END_DATE, errors='coerce'), 'qty': num(r.RQTY),
                         'buy': '', 'reason': r.EOUT_REASON_GB.replace('', '미기재'), 'doc': r.EOUT_NO.astype(str)})

    s = pd.DataFrame(rows(pick('상품매출')))
    sales = pd.DataFrame({'isbn': s.EAN2.astype(str), 'month': pd.PeriodIndex(s['__month'], freq='M'),
                          'store': 0.0, 'online': num(s.ORD_CNT).fillna(0), 'interpark': 0.0, 'corp': 0.0})
    sales = sales.groupby(['isbn', 'month'], as_index=False).sum()
    sales['total'] = sales[['store', 'online', 'interpark', 'corp']].sum(axis=1)
    write_reader_files(s, src.parent / '고객성향')
    return books, orders, rtgd, sales, hashes


def write_reader_files(s, folder):
    """월별 매출의 구매자 수치를 교보 고객성향과 같은 6개월 구간 형식으로 바꾼다."""
    folder.mkdir(exist_ok=True)
    s = s.assign(month=pd.PeriodIndex(s['__month'], freq='M'))
    last = pd.Period('2026-09', 'M')  # 2026-10은 1~3일만 있어 구매자 구간에서 뺀다
    s = s[s.month <= last]
    windows, cur = [], pd.Period('2021-10', 'M')  # 교보 고객성향과 같은 4~9월 / 10~3월 구간
    while cur <= last:
        windows.append((cur, cur + 5))
        cur += 6
    for a, b in windows:
        part = s[(s.month >= a) & (s.month <= b)]
        if part.empty:
            continue
        out = []
        for isbn, g in part.groupby(part.EAN2.astype(str)):
            total = num(g.ORD_CNT).sum()
            male, female = num(g.MALE).sum(), num(g.FEMALE).sum()
            gender = {'남자': int(male), '여자': int(female), '기타': int(max(total - male - female, 0))}
            ages = {AGE[k]: int(num(g[k]).sum()) for k in AGE if num(g[k]).sum() > 0}
            unknown_age = int(max(total - sum(ages.values()), 0))
            if unknown_age:
                ages['기타'] = unknown_age
            region = {REGION[k]: int(num(g[k]).sum()) for k in REGION if num(g[k]).sum() > 0}
            out.append(dict(isbn=isbn, status='ok', gender={k: v for k, v in gender.items() if v}, age_total=ages, region=region))
        period = [str(a.start_time.date()), str(min(b.end_time.date(), pd.Timestamp('2026-10-03').date()))]
        (folder / f'고객성향_{a}_{b}.json').write_text(json.dumps(dict(
            menu='상품매출관리 월별 → 6개월 구간 변환', period=period, unit='권',
            note='기타(성별) = 판매수량 − 남 − 여. 원자료의 NEUTRAL·NA는 합이 판매수량과 맞지 않아 쓰지 않았다. 연령 기타 = 판매수량 − 연령 확인분',
            rows=out), ensure_ascii=False), encoding='utf8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=Path, default=SRC)
    p.add_argument('--output', type=Path, default=ROOT.parent / '결과')
    args = p.parse_args()
    common.STORE.update(name='예스24', source='예스24 SCM (2026-10-04 수집)', event='발주', reader='예스24 구매자 집계')
    kyobo.load = load
    args.output.mkdir(parents=True, exist_ok=True)
    kyobo.run(args)


if __name__ == '__main__':
    main()
