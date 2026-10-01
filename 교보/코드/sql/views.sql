-- 품질 플래그 → 정제(추가 품질 점검 적용 조건) → 회사용 집계 뷰.
DROP VIEW IF EXISTS v_isbn_ok;
DROP VIEW IF EXISTS v_receipts_flagged;
DROP VIEW IF EXISTS v_returns_flagged;
DROP VIEW IF EXISTS v_receipts_clean;
DROP VIEW IF EXISTS v_returns_clean;
DROP VIEW IF EXISTS v_sales_clean;
DROP VIEW IF EXISTS v_books_clean;
DROP VIEW IF EXISTS v_monthly_kpi;
DROP VIEW IF EXISTS v_return_reasons;
DROP VIEW IF EXISTS v_series_sales_12m;
DROP VIEW IF EXISTS v_reader_period;

-- ISBN-13: 978/979로 시작하는 13자리 숫자이고 체크 숫자가 맞는가
CREATE VIEW v_isbn_ok AS
SELECT isbn,
       (isbn GLOB '97[89][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'
        AND (10 - ((substr(isbn,1,1) + 3*substr(isbn,2,1) + substr(isbn,3,1) + 3*substr(isbn,4,1)
                  + substr(isbn,5,1) + 3*substr(isbn,6,1) + substr(isbn,7,1) + 3*substr(isbn,8,1)
                  + substr(isbn,9,1) + 3*substr(isbn,10,1) + substr(isbn,11,1) + 3*substr(isbn,12,1)) % 10)) % 10
            = substr(isbn,13,1) + 0) AS ok
FROM (SELECT isbn FROM receipts UNION SELECT isbn FROM returns UNION SELECT isbn FROM sales_monthly UNION SELECT isbn FROM books);

CREATE VIEW v_receipts_flagged AS
SELECT r.rowid AS rid, r.*,
       r.date IS NULL                                        AS date_invalid,
       COALESCE(r.qty, 0) <= 0                               AS qty_nonpositive,
       NOT COALESCE(k.ok, 0)                                 AS isbn_invalid,
       r.isbn NOT IN (SELECT isbn FROM books)                AS isbn_not_in_books,
       ROW_NUMBER() OVER (PARTITION BY r.isbn, r.date, r.qty, r.buy, r.center, r.doc ORDER BY r.rowid) > 1 AS exact_duplicate
FROM receipts r LEFT JOIN v_isbn_ok k ON k.isbn = r.isbn;

CREATE VIEW v_returns_flagged AS
SELECT t.rowid AS rid, t.*,
       t.date IS NULL                                        AS date_invalid,
       COALESCE(t.qty, 0) <= 0                               AS qty_nonpositive,
       NOT COALESCE(k.ok, 0)                                 AS isbn_invalid,
       t.isbn NOT IN (SELECT isbn FROM books)                AS isbn_not_in_books,
       ROW_NUMBER() OVER (PARTITION BY t.isbn, t.date, t.qty, t.buy, t.reason, t.doc ORDER BY t.rowid) > 1 AS exact_duplicate
FROM returns t LEFT JOIN v_isbn_ok k ON k.isbn = t.isbn;

-- 추가 품질 점검 적용(full) 조건의 분석용 자료
CREATE VIEW v_receipts_clean AS
SELECT isbn, date, qty, buy, center, doc FROM v_receipts_flagged
WHERE NOT (date_invalid OR qty_nonpositive OR isbn_invalid OR isbn_not_in_books OR exact_duplicate);

CREATE VIEW v_returns_clean AS
SELECT isbn, date, qty, buy, reason, doc FROM v_returns_flagged
WHERE NOT (date_invalid OR qty_nonpositive OR isbn_invalid OR isbn_not_in_books OR exact_duplicate);

CREATE VIEW v_sales_clean AS
SELECT * FROM sales_monthly WHERE isbn IN (SELECT isbn FROM books);

CREATE VIEW v_books_clean AS
SELECT * FROM books WHERE pub IS NOT NULL;

-- 회사용: 월간 지표 (입하 = 교보 재주문)
CREATE VIEW v_monthly_kpi AS
WITH months AS (
    SELECT substr(date,1,7) AS month FROM v_receipts_clean
    UNION SELECT substr(date,1,7) FROM v_returns_clean
    UNION SELECT month FROM v_sales_clean
), r AS (SELECT substr(date,1,7) AS month, COUNT(*) AS orders, SUM(qty) AS rcvd_qty, COUNT(DISTINCT isbn) AS ordered_titles
         FROM v_receipts_clean GROUP BY 1),
   t AS (SELECT substr(date,1,7) AS month, SUM(qty) AS returned_qty FROM v_returns_clean GROUP BY 1),
   s AS (SELECT month, SUM(total) AS sold_qty, COUNT(DISTINCT CASE WHEN total > 0 THEN isbn END) AS selling_titles
         FROM v_sales_clean GROUP BY 1)
SELECT m.month,
       COALESCE(r.orders, 0)         AS reorders,
       COALESCE(r.rcvd_qty, 0)       AS reorder_copies,
       COALESCE(r.ordered_titles, 0) AS reordered_titles,
       COALESCE(t.returned_qty, 0)   AS returned_copies,
       COALESCE(s.sold_qty, 0)       AS sold_copies,
       COALESCE(s.selling_titles, 0) AS selling_titles,
       ROUND(COALESCE(t.returned_qty, 0) * 1.0 / NULLIF(r.rcvd_qty, 0), 3) AS return_to_receipt
FROM months m LEFT JOIN r USING (month) LEFT JOIN t USING (month) LEFT JOIN s USING (month)
ORDER BY m.month;

-- 회사용: 반품 사유별 비중
CREATE VIEW v_return_reasons AS
SELECT reason, SUM(qty) AS copies, ROUND(SUM(qty) * 1.0 / (SELECT SUM(qty) FROM v_returns_clean), 3) AS share
FROM v_returns_clean GROUP BY reason ORDER BY copies DESC;

-- 회사용: 시리즈(도서명 첫 단어, 'NEW' 제외)별 최근 12개월 판매
CREATE VIEW v_series_sales_12m AS
WITH b AS (
    SELECT isbn, CASE WHEN upper(substr(title,1,4)) = 'NEW ' THEN trim(substr(title,5)) ELSE trim(title) END AS t FROM books
), last AS (SELECT MAX(month) AS m FROM v_sales_clean)
SELECT substr(b.t, 1, instr(b.t || ' ', ' ') - 1) AS series,
       COUNT(DISTINCT b.isbn) AS titles,
       SUM(s.total)           AS sold_12m
FROM b JOIN v_sales_clean s ON s.isbn = b.isbn, last
WHERE s.month > strftime('%Y-%m', date(last.m || '-01', '-12 months'))
GROUP BY 1 HAVING COUNT(DISTINCT b.isbn) >= 2
ORDER BY sold_12m DESC;

-- 회사용: 6개월 구간별 구매자 구성
CREATE VIEW v_reader_period AS
SELECT period,
       COUNT(DISTINCT isbn) AS titles,
       SUM(CASE WHEN dimension = 'gender' THEN copies ELSE 0 END) AS copies,
       SUM(CASE WHEN dimension = 'gender' AND category IN ('남자','여자') THEN copies ELSE 0 END) AS known_gender,
       ROUND(SUM(CASE WHEN dimension = 'gender' AND category = '여자' THEN copies ELSE 0 END) * 1.0
             / NULLIF(SUM(CASE WHEN dimension = 'gender' AND category IN ('남자','여자') THEN copies ELSE 0 END), 0), 3) AS female_share,
       ROUND(SUM(CASE WHEN dimension = 'age_total' AND category GLOB '4[0-9]~*' THEN copies ELSE 0 END) * 1.0
             / NULLIF(SUM(CASE WHEN dimension = 'age_total' AND category GLOB '[0-9]*' THEN copies ELSE 0 END), 0), 3) AS age40s_share
FROM reader_profile GROUP BY period ORDER BY period;
