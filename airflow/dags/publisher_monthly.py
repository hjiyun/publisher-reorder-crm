"""출판사 SCM 월 배치 DAG.

매월 1일 09:00(KST)에 서점 원자료 적재 → 품질 점검 → 기준일 변수 → 월간 지표 저장을 순서대로 실행한다.
분석 코드는 새로 짜지 않고 마린북스/교보/코드/sql_layer.py 의 함수를 그대로 부른다.
"""
import os
import sqlite3
import sys

import pandas as pd
import pendulum
from airflow import DAG
from airflow.decorators import task

CODE = '/opt/airflow/marinebooks/교보/코드'       # docker-compose.yaml 에서 마운트한 경로
OUT = '/opt/airflow/marinebooks/교보/결과/airflow'  # 실행 날짜별 결과 폴더


def _sql_layer():
    if CODE not in sys.path:
        sys.path.insert(0, CODE)
    import sql_layer
    return sql_layer


def _out_dir(ds):
    path = f'{OUT}/{ds}'
    os.makedirs(path, exist_ok=True)
    return path


with DAG(
    dag_id='publisher_monthly',
    schedule='0 9 1 * *',                      # 매월 1일 09:00
    start_date=pendulum.datetime(2026, 1, 1, tz='Asia/Seoul'),
    catchup=False,
    max_active_runs=1,
    default_args={'retries': 2, 'retry_delay': pendulum.duration(minutes=5)},
    tags=['publisher', 'etl'],
) as dag:

    @task
    def load_to_sqlite():
        """원자료 읽기 + 파일 해시 기록 + SQLite 재적재(테이블을 지우고 다시 만들어 몇 번 돌려도 같은 결과)."""
        sl = _sql_layer()
        con = sl.build()
        files = con.execute('SELECT COUNT(*) FROM load_log').fetchone()[0]
        con.close()
        return files                            # XCom: 적재한 원본 파일 수

    @task
    def quality_checks(ds=None):
        """quality_checks.sql 규칙을 하나씩 실행하고 위반 건수를 CSV로 저장."""
        sl = _sql_layer()
        con = sqlite3.connect(sl.DB)
        checks = pd.DataFrame(sl.run_checks(con))
        con.close()
        checks.to_csv(f'{_out_dir(ds)}/quality_checks.csv', index=False, encoding='utf-8-sig')
        return int(checks['violations'].sum())  # XCom: 전체 위반 건수

    @task
    def build_features(ds=None, data_interval_end=None):
        """실행 월 1일을 기준일로 잡고, 기준일 이전 자료만 쓰는 도서별 변수를 만든다."""
        sl = _sql_layer()
        cutoff = data_interval_end.in_timezone('Asia/Seoul').start_of('month').date()
        con = sqlite3.connect(sl.DB)
        df = sl.features(con, cutoff)
        con.close()
        df.to_csv(f'{_out_dir(ds)}/features_{cutoff}.csv', index=False, encoding='utf-8-sig')
        return len(df)                          # XCom: 변수를 만든 도서 수

    @task
    def export_marts(ds=None):
        """출판사용 월간 지표 뷰를 CSV로 저장."""
        sl = _sql_layer()
        con = sqlite3.connect(sl.DB)
        for view in sl.MARTS:
            pd.read_sql_query(f'SELECT * FROM {view}', con).to_csv(
                f'{_out_dir(ds)}/{view}.csv', index=False, encoding='utf-8-sig')
        con.close()

    load_to_sqlite() >> quality_checks() >> build_features() >> export_marts()
