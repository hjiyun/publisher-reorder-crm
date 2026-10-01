-- 교보 SCM 자료 테이블. 원본(JSON·xls)을 Python이 형 변환만 해서 그대로 넣는다(정제는 뷰에서 한다).
-- 날짜는 'YYYY-MM-DD', 월은 'YYYY-MM' 문자열. 해석할 수 없는 날짜는 NULL.
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS receipts;
DROP TABLE IF EXISTS returns;
DROP TABLE IF EXISTS sales_monthly;
DROP TABLE IF EXISTS reader_profile;
DROP TABLE IF EXISTS load_log;

CREATE TABLE books (          -- 도서정보
    isbn   TEXT,
    title  TEXT,
    pub    TEXT,               -- 출판일
    price  REAL,               -- 정가
    genre  TEXT,               -- 분야
    status TEXT                -- 상품상태 코드
);
CREATE TABLE receipts (       -- 입하 상세 = 교보가 주문해 받은 기록(재주문)
    isbn   TEXT,
    date   TEXT,               -- 입하일
    qty    REAL,
    buy    TEXT,               -- 매입구분(위탁/일시)
    center TEXT,               -- 입하처
    doc    TEXT                -- 입하번호
);
CREATE TABLE returns (        -- 반품 상세
    isbn   TEXT,
    date   TEXT,               -- 반품일
    qty    REAL,
    buy    TEXT,
    reason TEXT,               -- 반품 사유
    doc    TEXT                -- 반품번호
);
CREATE TABLE sales_monthly (  -- 월별 도서 판매
    isbn      TEXT,
    month     TEXT,
    store     REAL,            -- 영업점
    online    REAL,
    interpark REAL,
    corp      REAL,            -- 법인
    total     REAL
);
CREATE TABLE reader_profile ( -- 상품별 고객성향(교보 구매자 집계), 긴 형식
    period    TEXT,            -- 'YYYY-MM-DD~YYYY-MM-DD'
    isbn      TEXT,
    dimension TEXT,            -- gender / age_total / region
    category  TEXT,            -- 남자, 40~44, 서울특별시 ...
    copies    INTEGER
);
CREATE TABLE load_log (       -- 원본 파일 해시(추적성)
    file   TEXT,
    sha256 TEXT,
    loaded TEXT
);
CREATE INDEX ix_receipts ON receipts(isbn, date);
CREATE INDEX ix_returns  ON returns(isbn, date);
CREATE INDEX ix_sales    ON sales_monthly(isbn, month);
CREATE INDEX ix_reader   ON reader_profile(period, isbn);
