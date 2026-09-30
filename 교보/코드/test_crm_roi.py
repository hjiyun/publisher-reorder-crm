import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import crm_roi as c


class CrmRoiTest(unittest.TestCase):
    def setUp(self):
        self.books = pd.DataFrame({'isbn': ['a', 'b', 'c'], 'title': ['가나다 A', '가나다 B', 'NEW 라마 C'], 'genre': ['문학', '문학', '인문'],
                                   'price': [10000.0, 10000.0, 20000.0]})
        self.rcvd = pd.DataFrame({'isbn': ['a', 'a', 'b', 'c'], 'qty': [10, 30, 10, 10], 'rate': [60, 70, 65, 60]})
        months = [pd.Period(m, 'M') for m in ('2026-04', '2026-05', '2026-06', '2026-07', '2026-08', '2026-09')]
        self.sales = pd.DataFrame({'isbn': ['a'] * 6 + ['b'] * 6, 'month': months * 2, 'total': [10] * 6 + [0, 0, 0, 0, 0, 6]})

    def test_breakeven_margin_and_allowance(self):
        df, s = c.breakeven(self.books, self.rcvd, self.sales)
        a = df.set_index('isbn').loc['a']
        # 가중 공급율 67.5% → 6,750 − 10,000×(0.20+0.10) − 300 = 3,450원
        self.assertAlmostEqual(a.supply_rate, 67.5)
        self.assertAlmostEqual(a.margin, 3450)
        self.assertAlmostEqual(a['uplift_100000'], 100000 / 3450 / 10)
        self.assertAlmostEqual(a['allow_0.3'], 0.3 * 10 * 3450)
        self.assertTrue(np.isnan(df.set_index('isbn').loc['c', 'uplift_100000']))  # 판매 없음 → 계산 불가
        self.assertEqual(s['series'][0]['series'], '가나다')

    def test_reader_frame_persona(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / '고객성향').mkdir()
            (Path(d) / '원본').mkdir()
            rows = [dict(isbn='a', gender={'남자': 1, '여자': 9, '기타': 5}, age_total={'40~44': 8, '35~39': 2, '기타': 5}, region={'서울특별시': 15}),
                    dict(isbn='b', gender={'기타': 2}, age_total={'기타': 2}, region={})]
            (Path(d) / '고객성향' / 'x.json').write_text(json.dumps({'period': ['2026-04-01', '2026-09-27'], 'rows': rows}), encoding='utf8')
            df = c.reader_frame(Path(d) / '원본').set_index('isbn')
        self.assertEqual(df.loc['a', 'persona'], '40대 여성 중심')
        self.assertEqual(df.loc['b', 'persona'], '표본 부족')
        self.assertEqual((df.loc['a', 'age_40대'], df.loc['a', 'age_30대'], df.loc['a', 'copies']), (8, 2, 15))

    def test_experiment_balanced_within_strata(self):
        be = pd.DataFrame({'isbn': list('abcdefgh'), 'title': list('abcdefgh'), 'genre': ['문학'] * 4 + ['인문'] * 4,
                           'monthly_sales': [10, 12, 3, 4, 20, 22, 5, 6], 'uplift_100000': np.linspace(.5, 2, 8)})
        sales = pd.DataFrame({'isbn': list('abcdefgh') * 3, 'month': [pd.Period(m, 'M') for m in ('2026-07',) * 8 + ('2026-08',) * 8 + ('2026-09',) * 8],
                              'total': np.arange(24) % 7})
        ex = c.experiment(be, sales, n=8)
        groups = pd.DataFrame(ex['rows']).groupby('stratum').group.value_counts().unstack()
        self.assertTrue((groups['판촉'] == groups['비교']).all())
        self.assertEqual(ex['per_arm'], 4)


if __name__ == '__main__':
    unittest.main()
