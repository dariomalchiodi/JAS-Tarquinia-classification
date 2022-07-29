import unittest

import numpy as np
import pandas as pd

from tarquinia.model_selection import MeasureStratifiedKFold
from tarquinia.model_selection import FragmentStratifiedKFold

class TestMeasureStratifiedKFold(unittest.TestCase):

    def setUp(self):
        self.X = pd.read_csv('data/data-tarquinia-latest.csv',
                             sep='\t', decimal=',', index_col='id')

    def test_init(self):
        with self.assertRaises(ValueError):
            MeasureStratifiedKFold(-3)

    def test_repr(self):
        f = MeasureStratifiedKFold()
        self.assertEqual(repr(f), 'MeasureStratifiedKFold()')

        f = MeasureStratifiedKFold(n_splits=3)
        self.assertEqual(repr(f), 'MeasureStratifiedKFold()')

        f = MeasureStratifiedKFold(n_splits=5)
        self.assertEqual(repr(f), 'MeasureStratifiedKFold(n_splits=5)')

        f = MeasureStratifiedKFold(shuffle=True)
        self.assertEqual(repr(f), 'MeasureStratifiedKFold(shuffle=True)')

        f = MeasureStratifiedKFold(shuffle=False)
        self.assertEqual(repr(f), 'MeasureStratifiedKFold()')

        f = MeasureStratifiedKFold(random_state=42)
        self.assertEqual(repr(f), 'MeasureStratifiedKFold(random_state=42)')

        f = MeasureStratifiedKFold(random_state=None)
        self.assertEqual(repr(f), 'MeasureStratifiedKFold()')

        f = MeasureStratifiedKFold(n_splits=3, shuffle=True)
        self.assertEqual(repr(f), 'MeasureStratifiedKFold(shuffle=True)')

        f = MeasureStratifiedKFold(n_splits=5, shuffle=True)
        self.assertEqual(repr(f),
                         'MeasureStratifiedKFold(n_splits=5, shuffle=True)')

    def test_split(self):
        f = MeasureStratifiedKFold()
        split = f.split(self.X)
        self.assertEqual(len(list(split)), 3)

        for n_splits in (2, 5, 7, 9):
            f = MeasureStratifiedKFold(n_splits)
            split = f.split(self.X)
            self.assertEqual(len(list(split)), n_splits)

    def test_intersection(self):
        for n_splits in (2, 5, 7, 9):
            f = MeasureStratifiedKFold(n_splits)
            for train, test in f.split(self.X):
                self.assertFalse(np.intersect1d(train, test))
                print(len(train), len(test))
    
    def test_equisize(self):
        for n_splits in (2, 5, 7, 9):
            f = MeasureStratifiedKFold(n_splits)
            train_len = []
            test_len = []
            for train, test in f.split(self.X):
                train_len.append(len(train))
                test_len.append(len(test))
            self.assertLessEqual(np.std(train_len), 1)
            self.assertLessEqual(np.std(test_len), 1)


class TestFragmentStratifiedKFold(unittest.TestCase):

    def setUp(self):
        self.X = pd.read_csv('data/data-tarquinia-v1.csv',
                             sep='\t', decimal=',', index_col='id')

    def test_init(self):
        with self.assertRaises(ValueError):
            FragmentStratifiedKFold(-3)

    def test_repr(self):
        f = FragmentStratifiedKFold()
        self.assertEqual(repr(f), 'FragmentStratifiedKFold()')

        f = FragmentStratifiedKFold(n_splits=3)
        self.assertEqual(repr(f), 'FragmentStratifiedKFold()')

        f = FragmentStratifiedKFold(n_splits=5)
        self.assertEqual(repr(f), 'FragmentStratifiedKFold(n_splits=5)')

        f = FragmentStratifiedKFold(shuffle=True)
        self.assertEqual(repr(f), 'FragmentStratifiedKFold(shuffle=True)')

        f = FragmentStratifiedKFold(shuffle=False)
        self.assertEqual(repr(f), 'FragmentStratifiedKFold()')

        f = FragmentStratifiedKFold(random_state=42)
        self.assertEqual(repr(f), 'FragmentStratifiedKFold(random_state=42)')

        f = FragmentStratifiedKFold(random_state=None)
        self.assertEqual(repr(f), 'FragmentStratifiedKFold()')

        f = FragmentStratifiedKFold(n_splits=3, shuffle=True)
        self.assertEqual(repr(f), 'FragmentStratifiedKFold(shuffle=True)')

        f = FragmentStratifiedKFold(n_splits=5, shuffle=True)
        self.assertEqual(repr(f),
                         'FragmentStratifiedKFold(n_splits=5, shuffle=True)')

    def test_split(self):
        f = FragmentStratifiedKFold()
        split = f.split(self.X)
        self.assertEqual(len(list(split)), 3)

        for n_splits in (2, 5, 7, 9):
            f = FragmentStratifiedKFold(n_splits)
            split = f.split(self.X)
            self.assertEqual(len(list(split)), n_splits)

    def test_intersection(self):
        for n_splits in (2, 5, 7, 9):
            f = FragmentStratifiedKFold(n_splits)
            for train, test in f.split(self.X):
                self.assertFalse(np.intersect1d(train, test))
                print(len(train), len(test))
    
    def test_equisize(self):
        for n_splits in (2, 5, 7, 9):
            f = FragmentStratifiedKFold(n_splits)
            train_len = []
            test_len = []
            for train, test in f.split(self.X, num_samples=2):
                train_len.append(len(train))
                test_len.append(len(test))
            self.assertLessEqual(np.std(train_len), 1)
            self.assertLessEqual(np.std(test_len), 1)

    def test_sampling(self):
        for n_splits in (2, 5, 7, 9):
            f = FragmentStratifiedKFold(n_splits=n_splits)
            fragments = f._get_fragments(self.X)
            for train, test in f.split(self.X, num_samples=2):
                self.assertEqual(len(train) + len(test), 2 * len(fragments))


            for train, test in f.split(self.X, num_samples=2):
                X_train = self.X.iloc[train]['FRAMMENTO']
                fragments = X_train.unique()
                for f in fragments:
                    assert(len(X_train[X_train==f]) == 2)
                
                X_test = self.X.iloc[train]['FRAMMENTO']
                fragments = X_test.unique()
                for f in fragments:
                    assert(len(X_test[X_test==f]) == 2)
                    
    