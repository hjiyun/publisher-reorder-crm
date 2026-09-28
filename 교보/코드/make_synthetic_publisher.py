"""출판사 데이터 형식을 흉내 낸 합성 데이터. 파이프라인 기능 검증용이며 실제 회사·고객을 나타내지 않는다.

데이터 요청서와 같은 파일명·한글 열 이름으로 data/publisher_SYNTHETIC/ 에 쓴다.
품질 점검이 잡아야 할 오류(누락 코드, ISBN 체크 숫자 오류, 중복 행, 연결 안 된 반품 등)를 일부러 섞는다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / '거래처장부' / '합성_SYNTHETIC'
START, END = pd.Timestamp('2022-01-01'), pd.Timestamp('2024-12-31')
rng = np.random.default_rng(7)


def isbn(n):
    core = f'97911{n:07d}'
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(core))
    return core + str((10 - total % 10) % 10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    genres = ['문학', '인문', '경제경영', '자기계발', '아동', '학습']
    n_books, n_accounts = 240, 70
    books = pd.DataFrame({
        'ISBN': [isbn(90000 + i) for i in range(n_books)],
        '분야': rng.choice(genres, n_books),
        '저자코드': [f'AU{rng.integers(1, 150):03d}' for _ in range(n_books)],
        '시리즈': np.where(rng.random(n_books) < .2, [f'SR{rng.integers(1, 12):02d}' for _ in range(n_books)], ''),
        '출간일': pd.to_datetime('2020-01-01') + pd.to_timedelta(rng.integers(0, 1740, n_books), unit='D'),
        '정가': rng.choice([12000, 14000, 15000, 16800, 18000, 22000], n_books),
    })
    types = rng.choice(['온라인서점', '대형서점', '총판', '지역서점'], n_accounts, p=[.06, .24, .14, .56])
    accounts = pd.DataFrame({'거래처코드': [f'C{i:03d}' for i in range(n_accounts)], '거래처유형': types,
                             '지역': rng.choice(['서울', '경기', '부산', '대구', '광주', '대전'], n_accounts),
                             '거래시작일': (pd.Timestamp('2015-01-01') + pd.to_timedelta(rng.integers(0, 2500, n_accounts), unit='D')).date})
    size = rng.lognormal(0, .7, n_accounts) * np.select([types == '온라인서점', types == '대형서점', types == '총판'], [6, 2.5, 3], 1)
    returns_p = rng.uniform(.08, .35, n_accounts)
    taste = rng.dirichlet(np.ones(len(genres)) * 2, n_accounts)
    popularity = rng.lognormal(0, 1, n_books)
    decay_weeks = rng.uniform(20, 120, n_books)
    genre_index = books['분야'].map({g: i for i, g in enumerate(genres)}).to_numpy()
    weeks = pd.date_range(START, END, freq='W-MON')

    rows, doc = [], 0
    for a in range(n_accounts):
        for b in range(n_books):
            pub = books['출간일'].iloc[b]
            intensity = size[a] * popularity[b] * taste[a, genre_index[b]] * len(genres)
            age = ((weeks - pub).days / 7).to_numpy()
            rate = np.where(age >= 0, np.minimum(.6, .03 * intensity * np.exp(-age / decay_weeks[b])), 0)
            rate[(age >= 0) & (age < 2)] = min(.9, .25 * intensity)  # 출간 직후 초도 주문
            for w in np.flatnonzero(rng.random(len(weeks)) < rate):
                date = weeks[w] + pd.Timedelta(days=int(rng.integers(0, 5)))
                qty = int(rng.poisson(1 + size[a] * 2)) + 1
                doc += 1
                price = round(books['정가'].iloc[b] * .65)
                rows.append([f'S{doc:07d}', date, accounts['거래처코드'][a], '', books.ISBN[b], '출고', qty, price, qty * price, ''])
                if rng.random() < returns_p[a]:
                    back = pd.Timedelta(days=int(rng.integers(30, 150)))
                    if date + back <= END:
                        r = int(rng.integers(1, qty + 1))
                        sign = -1 if rng.random() < .3 else 1  # 반품 수량 부호가 섞여 들어온 경우
                        rows.append([f'R{doc:07d}', date + back, accounts['거래처코드'][a], '', books.ISBN[b], '반품', sign * r, price, sign * r * price, f'S{doc:07d}'])
    cols = ['전표번호', '일자', '거래처코드', '지점코드', 'ISBN', '구분', '수량', '단가', '금액', '원전표번호']
    tx = pd.DataFrame(rows, columns=cols)
    tx['지점코드'] = np.where(tx['거래처코드'].map(dict(zip(accounts['거래처코드'], types))) == '대형서점', 'BR01', '')

    # 품질 점검이 잡아야 할 오류를 일부러 섞는다
    n = len(tx)
    pick = lambda share: rng.choice(n, int(n * share), replace=False)
    tx.loc[pick(.004), '거래처코드'] = ''
    bad = pick(.003)
    tx.loc[bad, 'ISBN'] = tx.loc[bad, 'ISBN'].str[:-1] + ((tx.loc[bad, 'ISBN'].str[-1].astype(int) + 1) % 10).astype(str)
    tx.loc[pick(.01), '금액'] = (tx['금액'] * 1.1).round()
    returns = np.flatnonzero(tx['구분'].eq('반품'))
    tx.loc[rng.choice(returns, len(returns) // 50, replace=False), '원전표번호'] = ''
    tx.loc[pick(.001), '구분'] = '반품취소'
    tx = pd.concat([tx, tx.iloc[pick(.005)]], ignore_index=True).sort_values('일자', kind='stable')
    tx['일자'] = pd.to_datetime(tx['일자']).dt.strftime('%Y-%m-%d')

    sales = []
    for b in range(n_books):
        age = ((weeks - books['출간일'].iloc[b]).days / 7).to_numpy()
        sold = rng.poisson(np.where(age >= 0, popularity[b] * 40 * np.exp(-age / decay_weeks[b]), 0))
        sales += [[books.ISBN[b], weeks[i].strftime('%Y-%m-%d'), int(s)] for i, s in enumerate(sold) if s]

    books['출간일'] = books['출간일'].dt.strftime('%Y-%m-%d')
    tx.to_csv(OUT / '출고반품.csv', index=False, encoding='utf-8-sig')
    accounts.to_csv(OUT / '거래처.csv', index=False, encoding='utf-8-sig')
    books.to_csv(OUT / '도서.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(sales, columns=['ISBN', '주차시작일', '판매수량']).to_csv(OUT / '서점판매.csv', index=False, encoding='utf-8-sig')
    print(OUT, len(tx), 'rows')


if __name__ == '__main__':
    main()
