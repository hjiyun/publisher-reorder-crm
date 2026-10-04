import unittest

import numpy as np
import pandas as pd

import error_injection as e
import kyobo

ISBN = '9780306406157'


def toy(n=40):
    books = pd.DataFrame({'isbn': [ISBN], 'title': ['t'], 'pub': pd.to_datetime(['2021-01-01']), 'price': [15000.0], 'genre': ['문학'], 'status': ['001']})
    dates = pd.date_range('2022-01-01', periods=n, freq='7D')
    rcvd = pd.DataFrame({'isbn': ISBN, 'date': dates, 'qty': np.arange(1, n + 1), 'buy': '위탁', 'center': 'c', 'doc': [str(i) for i in range(n)], 'rate': 60})
    rtgd = rcvd.assign(qty=1, reason='r')[['isbn', 'date', 'qty', 'buy', 'reason', 'doc']]
    months = pd.period_range('2022-01', periods=n, freq='M')
    sales = pd.DataFrame({'isbn': ISBN, 'month': months, 'store': 1, 'online': 2, 'interpark': 0, 'corp': 0})
    sales['total'] = 3
    return books, rcvd, rtgd, sales


class ErrorInjectionTest(unittest.TestCase):
    def test_rate_zero_leaves_data_unchanged(self):
        books, rcvd, rtgd, sales = toy()
        _, r, t, s, log = e.inject(books, rcvd, rtgd, sales, 0.0, list(e.TYPES), 0)
        self.assertEqual(log, [])
        pd.testing.assert_frame_equal(r, rcvd)
        pd.testing.assert_frame_equal(s, sales)

    def test_counts_and_types_are_balanced(self):
        books, rcvd, rtgd, sales = toy(60)
        _, r, _, _, log = e.inject(books, rcvd, rtgd, sales, 0.2, list(e.TYPES), 1)
        events = [x for x in log if x['table'] == 'rcvd']
        self.assertEqual(len(events), 12)
        self.assertEqual({x['type'] for x in events}, set(e.TABLES['rcvd']))
        self.assertEqual(len(r), 60 + 2 - 2)  # 중복 2행 추가, 누락 2행 삭제

    def test_typo_breaks_check_digit(self):
        self.assertFalse(kyobo.isbn13_valid(pd.Series([e.typo(ISBN)])).iloc[0])

    def test_rule_catches_intended_types_only(self):
        books, rcvd, rtgd, sales = toy(60)
        b, r, t, s, log = e.inject(books, rcvd, rtgd, sales, 0.3, list(e.TYPES), 2)
        _, drop = kyobo.quality(b, r, t, s)
        data = {n: kyobo.clean(b, r, t, s, drop, f) for n, f in (('minimal', False), ('full', True))}
        det = pd.DataFrame(e.detection(log, data['minimal'], data['full']))
        caught = det[det.table == 'rcvd'].groupby('type').full.mean()
        for kind in ('qty_zero', 'duplicate', 'isbn_typo'):
            self.assertEqual(caught[kind], 1.0)
        for kind in ('qty_x10', 'date_shift', 'row_missing'):
            self.assertEqual(caught[kind], 0.0)
        self.assertFalse(det.minimal.any())

    def test_already_flagged_rows_are_not_chosen(self):
        books, rcvd, rtgd, sales = toy(20)
        rcvd.loc[:9, 'qty'] = 0  # 원래부터 수량 0
        _, _, _, _, log = e.inject(books, rcvd, rtgd, sales, 0.5, ['qty_x10'], 3)
        self.assertTrue(all(x['row'] >= 10 for x in log if x['table'] == 'rcvd'))


if __name__ == '__main__':
    unittest.main()
