-- LLM Agent와 검증기가 쓰는 읽기 전용 조회. 연구 DB(../결과/research.sqlite)를 mode=ro로 연다.
-- 머리줄 형식:  -- query: 이름

-- query: latest_run
SELECT run_id FROM runs ORDER BY created DESC LIMIT 1;

-- query: evidence_value
-- Agent가 인용한 근거 ID의 값을 DB에서 다시 읽는다. 값이 다르면 검증기가 '수치 불일치'로 표시한다.
SELECT value, n FROM metrics WHERE run_id = :run_id AND evidence_id = :evidence_id;

-- query: evidence_list
-- Agent에게 넘길 근거 목록(수치와 근거 ID만, 원본 행은 넘기지 않는다)
SELECT evidence_id, condition, scope, subject, metric, value, n FROM metrics WHERE run_id = :run_id ORDER BY evidence_id;
