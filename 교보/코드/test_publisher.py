import unittest

import numpy as np
import pandas as pd

import publisher as p

GOOD, BAD = '9780306406157', '9780306406158'


def toy():
    tx = pd.DataFrame([
        # doc_id, date, account, isbn, kind, qty, unit_price, amount, orig_doc
        ('S1', '2023-01-10', 'A', GOOD, 'out', 10, 1000, 10000, None),
        ('S2', '2023-03-01', 'A', GOOD, 'out', 5, 1000, 5000, None),
        ('R1', '2023-04-01', 'A', GOOD, 'return', -3, 1000, -3000, 'S1'),
        ('R2', '2023-04-02', 'A', GOOD, 'return', 20, 1000, 20000, 'S2'),
        ('R3', '2023-04-03', 'A', GOOD, 'return', 1, 1000, 1000, None),
        ('S3', '2023-02-01', None, GOOD, 'out', 1, 1000, 1000, None),
        ('S4', '2023-02-01', 'B', BAD, 'out', 1, 1000, 1000, None),
        ('S5', '2023-02-01', 'B', GOOD, 'out', -2, 1000, 1500, None),
        ('S6', '2023-07-01', 'A', GOOD, 'out', 2, 1000, 2000, None),
    ], columns=['doc_id', 'date', 'account', 'isbn', 'kind', 'qty', 'unit_price', 'amount', 'orig_doc'])
    tx['date'] = pd.to_datetime(tx.date)
    tx['value'] = tx.amount.abs()
    books = pd.DataFrame({'isbn': [GOOD], 'genre': ['문학'], 'pub_date': pd.to_datetime(['2022-12-01']), 'list_price': [15000.0]})
    accounts = pd.DataFrame({'account': ['A', 'B'], 'account_type': ['총판', '지역서점']})
    return tx, books, accounts


class PublisherTest(unittest.TestCase):
    def test_isbn_checksum(self):
        s = pd.Series([GOOD, BAD, '123', None])
        self.assertEqual(p.isbn13_valid(s).tolist(), [True, False, False, False])

    def test_quality_flags(self):
        tx, books, accounts = toy()
        f = p.quality_flags(tx, books, accounts)
        self.assertEqual(tx.doc_id[f.account_missing].tolist(), ['S3'])
        self.assertEqual(tx.doc_id[f.isbn_invalid].tolist(), ['S4'])
        self.assertEqual(tx.doc_id[f.return_unlinked].tolist(), ['R3'])
        self.assertEqual(tx.doc_id[f.return_exceeds_shipped].tolist(), ['R2', 'R3'])
        self.assertEqual(tx.doc_id[f.sign_inconsistent].tolist(), ['S5'])
        self.assertEqual(tx.doc_id[f.amount_mismatch].tolist(), ['S5'])

    def test_full_clean_takes_abs_and_drops_bad_returns(self):
        tx, books, accounts = toy()
        full = p.clean(tx, p.quality_flags(tx, books, accounts), full=True)
        self.assertEqual(full.doc_id.tolist(), ['S1', 'S2', 'R1', 'S6'])
        self.assertTrue((full.qty > 0).all())
        self.assertEqual(len(p.clean(tx, p.quality_flags(tx, books, accounts), full=False)), 8)

    def test_snapshot_ignores_future_and_label(self):
        tx, books, accounts = toy()
        tx = p.clean(tx, p.quality_flags(tx, books, accounts), full=True)
        cutoff = pd.Timestamp('2023-06-01')
        before = p.snapshot(tx, books, accounts, None, cutoff)
        future = tx.copy()
        future.loc[future.date >= cutoff, 'qty'] = 999
        pd.testing.assert_frame_equal(before, p.snapshot(future, books, accounts, None, cutoff))
        row = before.iloc[0]
        self.assertEqual((row.orders, row.qty, row.return_qty, row.recency_days), (2, 15, 3, 92))
        self.assertEqual(p.reorders(tx, cutoff), {('A', GOOD)})
        self.assertEqual(p.reorders(tx, pd.Timestamp('2023-04-02')), set())  # 구간 끝(7/1)은 포함하지 않는다
        self.assertEqual(p.reorders(tx, pd.Timestamp('2023-04-03')), {('A', GOOD)})
        self.assertEqual(p.reorders(tx, pd.Timestamp('2023-10-01')), set())

    def test_cutoffs_are_quarterly_and_ordered(self):
        tx = pd.DataFrame({'date': pd.to_datetime(['2022-01-01', '2024-12-31'])})
        c = p.make_cutoffs(tx)
        self.assertEqual(c['test'], ['2024-10-01'])
        self.assertEqual(c['val'], ['2024-07-01'])
        self.assertEqual(c['train'][0], '2023-01-01')
        with self.assertRaises(ValueError):
            p.make_cutoffs(pd.DataFrame({'date': pd.to_datetime(['2024-01-01', '2024-12-31'])}))


if __name__ == '__main__':
    unittest.main()
