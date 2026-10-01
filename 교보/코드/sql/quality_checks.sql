-- ISO/IEC 25012 관점을 참고한 품질 점검 규칙. 쿼리마다 (위반 건수, 대상 건수)를 돌려준다.
-- 머리줄 형식:  -- check: 자료|점검명|관점|규칙|처리

-- check: 입하|date_invalid|완전성|일자 누락·해석 불가|제외 (두 조건 공통)
SELECT SUM(date_invalid), COUNT(*) FROM v_receipts_flagged;
-- check: 입하|qty_nonpositive|일관성|수량 0 이하·누락|full: 제외
SELECT SUM(qty_nonpositive), COUNT(*) FROM v_receipts_flagged;
-- check: 입하|isbn_invalid|일관성|ISBN 형식·체크 숫자 오류|full: 제외
SELECT SUM(isbn_invalid), COUNT(*) FROM v_receipts_flagged;
-- check: 입하|isbn_not_in_books|일관성|도서정보에 없는 ISBN|full: 제외
SELECT SUM(isbn_not_in_books), COUNT(*) FROM v_receipts_flagged;
-- check: 입하|exact_duplicate|일관성|모든 열이 같은 중복 행|full: 제외
SELECT SUM(exact_duplicate), COUNT(*) FROM v_receipts_flagged;
-- check: 입하|before_publication|일관성|출간일 이전 입하|예약 입하일 수 있어 기록만 함
SELECT SUM(r.date < b.pub), COUNT(*) FROM receipts r LEFT JOIN books b ON b.isbn = r.isbn;

-- check: 반품|date_invalid|완전성|일자 누락·해석 불가|제외 (두 조건 공통)
SELECT SUM(date_invalid), COUNT(*) FROM v_returns_flagged;
-- check: 반품|qty_nonpositive|일관성|수량 0 이하·누락|full: 제외
SELECT SUM(qty_nonpositive), COUNT(*) FROM v_returns_flagged;
-- check: 반품|isbn_invalid|일관성|ISBN 형식·체크 숫자 오류|full: 제외
SELECT SUM(isbn_invalid), COUNT(*) FROM v_returns_flagged;
-- check: 반품|isbn_not_in_books|일관성|도서정보에 없는 ISBN|full: 제외
SELECT SUM(isbn_not_in_books), COUNT(*) FROM v_returns_flagged;
-- check: 반품|exact_duplicate|일관성|모든 열이 같은 중복 행|full: 제외
SELECT SUM(exact_duplicate), COUNT(*) FROM v_returns_flagged;
-- check: 반품|return_exceeds_received|일관성|누적 반품 > 누적 입하 (입하 기록 이전 공급분 가능)|기록만 함
WITH flow AS (
    SELECT isbn, date, qty AS in_qty, 0 AS out_qty, 0 AS src, rowid AS rid FROM receipts WHERE date IS NOT NULL
    UNION ALL
    SELECT isbn, date, 0, qty, 1, rowid FROM returns WHERE date IS NOT NULL
), cum AS (
    SELECT out_qty,
           SUM(in_qty)  OVER w AS cum_in,
           SUM(out_qty) OVER w AS cum_out
    FROM flow
    WINDOW w AS (PARTITION BY isbn ORDER BY date, out_qty, src, rid ROWS UNBOUNDED PRECEDING)
)
SELECT SUM(out_qty > 0 AND cum_out > cum_in), (SELECT COUNT(*) FROM returns) FROM cum;

-- check: 판매|month_missing|완전성|판매 월 누락 (첫 달~마지막 달 사이)|해당 월 판매 변수 결측
WITH RECURSIVE m(month) AS (
    SELECT MIN(month) FROM sales_monthly
    UNION ALL
    SELECT strftime('%Y-%m', date(month || '-01', '+1 month')) FROM m WHERE month < (SELECT MAX(month) FROM sales_monthly)
)
SELECT SUM(month NOT IN (SELECT month FROM sales_monthly)), COUNT(*) FROM m;
-- check: 판매|isbn_not_in_books|일관성|도서정보에 없는 ISBN의 판매 행|full: 제외
SELECT SUM(isbn NOT IN (SELECT isbn FROM books)), COUNT(*) FROM sales_monthly;
-- check: 판매|negative_sales|일관성|판매 수량 음수(반품 상계)|기록만 함
SELECT SUM(store < 0 OR online < 0 OR interpark < 0 OR corp < 0), COUNT(*) FROM sales_monthly;

-- check: 도서|pub_missing|완전성|출판일 누락|대상에서 제외 (두 조건 공통)
SELECT SUM(pub IS NULL), COUNT(*) FROM books;

-- check: 구매자|period_all_unknown|완전성|성별·연령이 전부 미기록인 6개월 구간|해석에서 제외
WITH p AS (SELECT period, SUM(CASE WHEN dimension = 'gender' AND category IN ('남자','여자') THEN copies ELSE 0 END) AS known
           FROM reader_profile GROUP BY period)
SELECT SUM(known = 0), COUNT(*) FROM p;
-- check: 구매자|bulk_purchase|일관성|한 연령 구간에 100권 이상이 몰린 도서(기관·대량 구매 의심)|개인 구매 분포로 해석하지 않음
WITH a AS (SELECT period, isbn, category, copies,
                  SUM(copies) OVER (PARTITION BY period, isbn) AS known_age
           FROM reader_profile WHERE dimension = 'age_total' AND category GLOB '[0-9]*')
SELECT SUM(copies >= 100 AND copies >= 0.5 * known_age),
       (SELECT COUNT(*) FROM (SELECT DISTINCT period, isbn FROM reader_profile)) FROM a;
