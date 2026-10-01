"""교보 SCM 자료의 SQLite 데이터 계층.

적재(형 변환)는 Python, 품질 점검·기준일 변수·회사용 집계는 sql/ 폴더의 SQL이 맡는다.

실행:  python sql_layer.py      → ../DB/kyobo.sqlite 생성, ../결과/SQL/<시각>/ 에 점검표·집계 CSV와 index.html
"""
import json
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd

import common
import kyobo

ROOT = Path(__file__).resolve().parent
SQL = ROOT / 'sql'
DB = ROOT.parent / 'DB' / 'kyobo.sqlite'
RESEARCH_DB = ROOT.parent / '결과' / 'research.sqlite'
MARTS = ['v_monthly_kpi', 'v_return_reasons', 'v_series_sales_12m', 'v_reader_period']


def _blocks(text, tag):
    """'-- tag: 머리줄' 로 나뉜 SQL 블록을 (머리줄, SQL) 목록으로 돌려준다."""
    parts = re.split(rf'^-- {tag}: *(.+)$', text, flags=re.M)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def reader_rows(folder):
    rows = []
    for path in sorted(folder.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf8'))
        period = '~'.join(payload.get('period', [path.stem, '']))
        for r in payload['rows']:
            for dim in ('gender', 'age_total', 'region'):
                rows += [(period, r['isbn'], dim, k, int(v)) for k, v in r.get(dim, {}).items()]
    return rows


def build(db=DB, src=kyobo.SRC):
    books, rcvd, rtgd, sales, hashes = kyobo.load(src)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript((SQL / 'schema.sql').read_text(encoding='utf8'))
    day = lambda s: s.dt.strftime('%Y-%m-%d').where(s.notna(), None)
    none = lambda df: df.astype(object).where(df.notna(), None)
    con.executemany('INSERT INTO books VALUES (?,?,?,?,?,?)', none(books.assign(pub=day(books.pub)))[
        ['isbn', 'title', 'pub', 'price', 'genre', 'status']].itertuples(index=False))
    con.executemany('INSERT INTO receipts VALUES (?,?,?,?,?,?)', none(rcvd.assign(date=day(rcvd.date)))[
        ['isbn', 'date', 'qty', 'buy', 'center', 'doc']].itertuples(index=False))
    con.executemany('INSERT INTO returns VALUES (?,?,?,?,?,?)', none(rtgd.assign(date=day(rtgd.date)))[
        ['isbn', 'date', 'qty', 'buy', 'reason', 'doc']].itertuples(index=False))
    con.executemany('INSERT INTO sales_monthly VALUES (?,?,?,?,?,?,?)', none(sales.assign(month=sales.month.astype(str)))[
        ['isbn', 'month', 'store', 'online', 'interpark', 'corp', 'total']].itertuples(index=False))
    con.executemany('INSERT INTO reader_profile VALUES (?,?,?,?,?)', reader_rows(src.parent / '고객성향'))
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    con.executemany('INSERT INTO load_log VALUES (?,?,?)', [(f, h, now) for f, h in hashes.items()])
    con.executescript((SQL / 'views.sql').read_text(encoding='utf8'))
    con.commit()
    return con


def run_checks(con):
    out = []
    for head, sql in _blocks((SQL / 'quality_checks.sql').read_text(encoding='utf8'), 'check'):
        table, check, dimension, rule, action = head.split('|')
        violations, n = con.execute(sql).fetchone()
        violations, n = int(violations or 0), int(n or 0)
        out.append(dict(table=table, check=check, dimension=dimension, rule=rule, violations=violations,
                        share=violations / n if n else 0.0, action=action))
    return out


def features(con, cutoff):
    sql = (SQL / 'features.sql').read_text(encoding='utf8')
    return pd.read_sql_query(sql, con, params={'cutoff': str(pd.Timestamp(cutoff).date())})


def read_only(name, db=RESEARCH_DB, **params):
    """agent_queries.sql의 이름 붙은 조회만 읽기 전용 연결로 실행한다(LLM Agent·검증기용)."""
    queries = dict(_blocks((SQL / 'agent_queries.sql').read_text(encoding='utf8'), 'query'))
    sql = queries[name]
    body = '\n'.join(l for l in sql.splitlines() if not l.strip().startswith('--')).strip()
    if not re.match(r'(?is)^(select|with)\b', body):
        raise ValueError('읽기 전용 조회만 허용한다')
    con = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
    try:
        return con.execute(body, params).fetchall()
    finally:
        con.close()


def main():
    started = time.time()
    con = build()
    checks = run_checks(con)
    out = ROOT.parent / '결과' / 'SQL' / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    pd.DataFrame(checks).to_csv(out / 'quality_checks.csv', index=False, encoding='utf-8-sig')
    marts = {}
    for view in MARTS:
        df = pd.read_sql_query(f'SELECT * FROM {view}', con)
        df.to_csv(out / f'{view}.csv', index=False, encoding='utf-8-sig')
        marts[view] = df
    pct, num = common.pct, lambda v: f'{v:,.0f}'
    qc = [('table', '자료', None), ('dimension', '관점', None), ('rule', '점검 규칙', None), ('violations', '위반', num), ('share', '비율', lambda v: pct(v, 2)), ('action', '처리', None)]
    kpi = marts['v_monthly_kpi'].tail(24).to_dict('records')
    kc = [('month', '월', None), ('reorders', '재주문 건', num), ('reorder_copies', '재주문 권', num), ('reordered_titles', '재주문 도서', num),
          ('returned_copies', '반품 권', num), ('sold_copies', '판매 권', num), ('return_to_receipt', '반품/입하', lambda v: '–' if v is None or v != v else pct(v))]
    rc = [('reason', '반품 사유', None), ('copies', '권', num), ('share', '비중', pct)]
    sc = [('series', '시리즈', None), ('titles', '도서', num), ('sold_12m', '최근 12개월 판매', num)]
    pc = [('period', '구간', None), ('titles', '도서', num), ('copies', '권', num), ('known_gender', '성별 확인', num),
          ('female_share', '여성', lambda v: '–' if v is None or v != v else pct(v)), ('age40s_share', '40대', lambda v: '–' if v is None or v != v else pct(v))]
    body = f'''<p class="tag">교보 SCM · SQL 데이터 계층 · {out.name}</p><h1>SQL 품질 점검과 회사용 월간 지표</h1>
<p class="note">DB: ../DB/kyobo.sqlite · 규칙: 코드/sql/quality_checks.sql · 집계: 코드/sql/views.sql</p>
<section><h2>품질 점검 (ISO/IEC 25012 참고)</h2>{common.table(checks, qc)}</section>
<section><h2>월간 지표 (최근 24개월)</h2><p class="note">재주문 = 교보 입하. 반품/입하는 같은 달 기준 비율.</p>{common.table(kpi, kc)}</section>
<section><h2>반품 사유</h2>{common.table(marts['v_return_reasons'].to_dict('records'), rc)}</section>
<section><h2>시리즈별 최근 12개월 판매</h2>{common.table(marts['v_series_sales_12m'].head(10).to_dict('records'), sc)}</section>
<section><h2>구간별 구매자 구성</h2>{common.table(marts['v_reader_period'].to_dict('records'), pc)}</section>'''
    (out / 'index.html').write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>교보 SQL 계층</title><style>{common.CSS}</style>{body}</html>', encoding='utf8')
    con.close()
    print(out / 'index.html', f'{time.time() - started:.0f}s')


if __name__ == '__main__':
    main()
