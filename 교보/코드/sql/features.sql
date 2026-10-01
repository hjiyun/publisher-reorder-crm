-- 기준일(:cutoff, 'YYYY-MM-DD')의 도서별 변수. 기준일 이전 기록만 읽는다(정보 누출 방지).
-- 대상: 기준일 이전 출간 + 직전 365일 안에 입하 또는 직전 12개월 안에 판매(월 합계 > 0)가 있는 도서.
-- kyobo.py의 snapshot()과 같은 값을 내도록 맞췄다(test_sql_layer.py가 대조한다). 분야 더미는 Python에서 붙인다.
WITH p AS (
    SELECT :cutoff AS c,
           strftime('%Y-%m', :cutoff) AS m,
           strftime('%Y-%m', date(:cutoff, 'start of month', '-1 month'))   AS m1,
           strftime('%Y-%m', date(:cutoff, 'start of month', '-3 months'))  AS m3,
           strftime('%Y-%m', date(:cutoff, 'start of month', '-12 months')) AS m12
), r AS (SELECT v.* FROM v_receipts_clean v, p WHERE v.date < p.c),
   t AS (SELECT v.* FROM v_returns_clean v, p WHERE v.date < p.c),
   s AS (SELECT v.* FROM v_sales_clean v, p WHERE v.month < p.m),
active AS (
    SELECT isbn FROM r, p WHERE r.date >= date(p.c, '-365 days')
    UNION
    SELECT isbn FROM s, p WHERE s.month >= p.m12 AND s.total > 0
), ra AS (
    SELECT isbn,
           SUM(CASE WHEN date >= date(p.c, '-30 days')  THEN qty ELSE 0 END) AS rcvd_qty_30d,
           SUM(CASE WHEN date >= date(p.c, '-30 days')  THEN 1   ELSE 0 END) AS rcvd_n_30d,
           SUM(CASE WHEN date >= date(p.c, '-90 days')  THEN qty ELSE 0 END) AS rcvd_qty_90d,
           SUM(CASE WHEN date >= date(p.c, '-90 days')  THEN 1   ELSE 0 END) AS rcvd_n_90d,
           SUM(CASE WHEN date >= date(p.c, '-365 days') THEN qty ELSE 0 END) AS rcvd_qty_365d,
           SUM(CASE WHEN date >= date(p.c, '-365 days') THEN 1   ELSE 0 END) AS rcvd_n_365d,
           SUM(CASE WHEN date >= date(p.c, '-365 days') AND buy = '위탁' THEN qty ELSE 0 END) AS wtak_365d,
           julianday(p.c) - julianday(MAX(date)) AS days_since_rcvd
    FROM r, p GROUP BY isbn
), ta AS (
    SELECT isbn,
           SUM(CASE WHEN date >= date(p.c, '-90 days')  THEN qty ELSE 0 END) AS rtgd_qty_90d,
           SUM(CASE WHEN date >= date(p.c, '-365 days') THEN qty ELSE 0 END) AS rtgd_qty_365d
    FROM t, p GROUP BY isbn
), sa AS (
    SELECT isbn,
           SUM(CASE WHEN month = p.m1   THEN total  ELSE 0 END) AS sale_1m,
           SUM(CASE WHEN month >= p.m3  THEN total  ELSE 0 END) AS sale_3m,
           SUM(CASE WHEN month >= p.m12 THEN total  ELSE 0 END) AS sale_12m,
           SUM(CASE WHEN month >= p.m3  THEN store  ELSE 0 END) AS sale_store_3m,
           SUM(CASE WHEN month >= p.m3  THEN online ELSE 0 END) AS sale_online_3m,
           SUM(CASE WHEN month >= p.m12 AND total > 0 THEN 1 ELSE 0 END) AS sale_months_12,
           MAX(CASE WHEN total > 0 THEN month END) AS last_sale_month
    FROM s, p GROUP BY isbn
)
SELECT b.isbn,
       (julianday(p.c) - julianday(b.pub)) / 30.4                    AS age_months,
       b.price                                                       AS price,
       COALESCE(ra.rcvd_qty_30d, 0)  AS rcvd_qty_30d,  COALESCE(ra.rcvd_n_30d, 0)  AS rcvd_n_30d,
       COALESCE(ra.rcvd_qty_90d, 0)  AS rcvd_qty_90d,  COALESCE(ra.rcvd_n_90d, 0)  AS rcvd_n_90d,
       COALESCE(ra.rcvd_qty_365d, 0) AS rcvd_qty_365d, COALESCE(ra.rcvd_n_365d, 0) AS rcvd_n_365d,
       ra.days_since_rcvd                                            AS days_since_rcvd,
       COALESCE(ta.rtgd_qty_90d, 0)  AS rtgd_qty_90d,
       COALESCE(ta.rtgd_qty_365d, 0) AS rtgd_qty_365d,
       COALESCE(ta.rtgd_qty_365d, 0) * 1.0 / NULLIF(ra.rcvd_qty_365d, 0) AS return_ratio_365d,
       COALESCE(ra.wtak_365d, 0) * 1.0 / NULLIF(ra.rcvd_qty_365d, 0)     AS wtak_share_365d,
       COALESCE(sa.sale_1m, 0)  AS sale_1m,
       COALESCE(sa.sale_3m, 0)  AS sale_3m,
       COALESCE(sa.sale_12m, 0) AS sale_12m,
       COALESCE(sa.sale_store_3m, 0)  AS sale_store_3m,
       COALESCE(sa.sale_online_3m, 0) AS sale_online_3m,
       COALESCE(sa.sale_months_12, 0) AS sale_months_12,
       CASE WHEN COALESCE(sa.sale_3m, 0) > 0 THEN sa.sale_1m / (sa.sale_3m / 3.0) END AS sale_trend,
       julianday(p.c) - julianday(date(sa.last_sale_month || '-01', '+1 month'))       AS days_since_sale,
       COALESCE(ra.rcvd_qty_365d, 0) - COALESCE(ta.rtgd_qty_365d, 0) - COALESCE(sa.sale_12m, 0) AS net_flow_365d
FROM v_books_clean b
CROSS JOIN p
JOIN active a        ON a.isbn = b.isbn
LEFT JOIN ra         ON ra.isbn = b.isbn
LEFT JOIN ta         ON ta.isbn = b.isbn
LEFT JOIN sa         ON sa.isbn = b.isbn
WHERE b.pub < p.c
ORDER BY b.isbn;
