import unittest

import pandas as pd

import kyobo as k

ISBN = '9780306406157'


def toy():
    books = pd.DataFrame({'isbn': [ISBN], 'title': ['t'], 'pub': pd.to_datetime(['2021-01-01']), 'price': [15000.0], 'genre': ['문학'], 'status': ['001']})
    rcvd = pd.DataFrame({'isbn': [ISBN] * 3, 'date': pd.to_datetime(['2022-01-10', '2022-03-05', '2022-04-15']), 'qty': [10, 5, 7],
                         'buy': ['위탁', '일시', '위탁'], 'center': ['파주센터'] * 3, 'doc': ['1', '2', '3']})
    rtgd = pd.DataFrame({'isbn': [ISBN], 'date': pd.to_datetime(['2022-02-20']), 'qty': [3], 'buy': ['위탁'], 'reason': ['과다재고'], 'doc': ['r1']})
    sales = pd.DataFrame({'isbn': [ISBN] * 3, 'month': [pd.Period(m, 'M') for m in ('2022-01', '2022-02', '2022-03')],
                          'store': [2, 1, 4], 'online': [1, 0, 2], 'interpark': [0, 0, 0], 'corp': [0, 0, 0]})
    sales['total'] = sales[['store', 'online', 'interpark', 'corp']].sum(axis=1)
    return books, rcvd, rtgd, sales


class KyoboTest(unittest.TestCase):
    def test_snapshot_uses_only_past(self):
        books, rcvd, rtgd, sales = toy()
        c = pd.Timestamp('2022-04-01')
        x = k.snapshot(books, rcvd, rtgd, sales, c).iloc[0]
        self.assertEqual((x.rcvd_qty_365d, x.rtgd_qty_365d, x.sale_3m, x.sale_1m), (15, 3, 10, 6))
        self.assertEqual(x.days_since_rcvd, 27)
        self.assertAlmostEqual(x.wtak_share_365d, 10 / 15)
        future = rcvd.copy()
        future.loc[future.date >= c, 'qty'] = 999
        pd.testing.assert_frame_equal(k.snapshot(books, rcvd, rtgd, sales, c), k.snapshot(books, future, rtgd, sales, c))

    def test_reorder_window_is_30_days(self):
        _, rcvd, _, _ = toy()
        self.assertEqual(k.reorder(rcvd, pd.Timestamp('2022-04-01')), {ISBN})
        self.assertEqual(k.reorder(rcvd, pd.Timestamp('2022-03-06')), set())  # 4/15는 30일 밖

    def test_monthly_pick_takes_20_percent_each_month(self):
        test = pd.DataFrame({'cutoff': ['a'] * 10 + ['b'] * 5})
        pick = k.monthly_pick(test, pd.Series(range(15)).to_numpy(float), .2)
        self.assertEqual(pick[:10].sum(), 2)
        self.assertEqual(pick[10:].sum(), 1)
        self.assertTrue(pick[8] and pick[9] and pick[14])

    def test_age_band(self):
        self.assertEqual([k.age_band(x) for x in ('20~24', '35~39', '55~59', '기타', '15~19')], ['20대', '30대', '50대 이상', '기타', '10대 이하'])


if __name__ == '__main__':
    unittest.main()
