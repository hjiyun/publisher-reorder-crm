"""SQL 계층이 Python(kyobo.py)과 같은 결과를 내는지 실제 교보 자료로 대조한다."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import kyobo
import sql_layer


class SqlLayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (kyobo.SRC / '입하상세').exists():
            raise unittest.SkipTest('교보 원본 자료가 없다')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.con = sql_layer.build(Path(cls.tmp.name) / 't.sqlite')
        cls.books, cls.rcvd, cls.rtgd, cls.sales, _ = kyobo.load(kyobo.SRC)
        cls.qrows, cls.drop = kyobo.quality(cls.books, cls.rcvd, cls.rtgd, cls.sales)
        cls.full = kyobo.clean(cls.books, cls.rcvd, cls.rtgd, cls.sales, cls.drop, True)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()

    def test_quality_counts_match_python(self):
        sql = {(r['table'], r['check']): r['violations'] for r in sql_layer.run_checks(self.con)}
        for r in self.qrows:
            self.assertEqual(sql[(r['table'], r['check'])], r['violations'], (r['table'], r['check']))
        self.assertIn(('구매자', 'period_all_unknown'), sql)

    def test_features_match_python_snapshot(self):
        cuts = kyobo.cutoffs(self.full[1], self.full[3])
        dates = [cuts['train'][0], cuts['train'][len(cuts['train']) // 2], cuts['val'][0], cuts['test'][-1]]
        for d in dates:
            py = kyobo.snapshot(*self.full, pd.Timestamp(d)).set_index('isbn').sort_index()
            sq = sql_layer.features(self.con, d).set_index('isbn').sort_index()
            self.assertEqual(list(py.index), list(sq.index), d)
            for col in sq.columns:
                a, b = py[col].to_numpy(float), sq[col].to_numpy(float)
                self.assertTrue(np.allclose(a, b, equal_nan=True), (d, col))

    def test_marts_are_consistent(self):
        kpi = pd.read_sql_query('SELECT * FROM v_monthly_kpi', self.con)
        self.assertEqual(kpi.reorder_copies.sum(), self.full[1].qty.sum())
        self.assertEqual(kpi.returned_copies.sum(), self.full[2].qty.sum())
        share = pd.read_sql_query('SELECT SUM(share) AS s FROM v_return_reasons', self.con).s[0]
        self.assertAlmostEqual(share, 1.0, places=2)

    def test_read_only_rejects_writes(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / 'r.sqlite'
            c = sqlite3.connect(db)
            c.executescript("CREATE TABLE runs(run_id, created); INSERT INTO runs VALUES ('r1','2026');"
                            "CREATE TABLE metrics(evidence_id, run_id, condition, scope, subject, metric, value, n);"
                            "INSERT INTO metrics VALUES ('e1','r1','full','test','boosting','precision20',0.52,579);")
            c.close()
            self.assertEqual(sql_layer.read_only('latest_run', db=db), [('r1',)])
            self.assertEqual(sql_layer.read_only('evidence_value', db=db, run_id='r1', evidence_id='e1'), [(0.52, 579)])
            ro = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
            with self.assertRaises(sqlite3.OperationalError):
                ro.execute("DELETE FROM runs")
            ro.close()


if __name__ == '__main__':
    unittest.main()
