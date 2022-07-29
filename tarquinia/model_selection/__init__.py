from doctest import ELLIPSIS_MARKER
import numpy as np
from sklearn.model_selection import StratifiedKFold

class MeasureStratifiedKFold:
    """K-Folds cross-validator with stratification on measures.

    Provides train/test indices to split data in train/test sets.
    Stratification is done on labels for all measures, thus it is equivalent to
    a standard cross-validation: folds preserve the fraction of samples for
    each class, but train and test set might contain different measures of a
    same fragment.
    
    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tarquinia.model_selection import MeasureStratifiedKFold
    >>> X = pd.DataFrame({'FRAMMENTO': [0, 0, 1, 1, 2, 2, 2, 3, 3, 3],
                          'feature': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          'PROVENIENZA': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})
    >>> mskf = MeasureStratifiedKFold(n_splits=2)
    >>> print(skf)
    MeasureStratifiedKFold(n_splits=2)
    >>> for train_index, test_index in mskf.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    TRAIN: [2 3 7 8 9] TEST: [0 1 4 5 6]
    
    Notes
    -----
    The implementation expects to work with datasets encoded as pandas
    dataframes, containing an attribute describing class labels (see
    documentation of the `split` method). Using other data structures such as
    numpy arrays is not allowed.
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        """Build an object of 
        """
        if n_splits < 2:
            raise ValueError(f'number of folds should be greater than 1 ('
                             f'provided n_splits={n_splits})')
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, label_col='PROVENIENZA'):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            Note that, unlike sklearn, the dataframe should include a column
            describing labels.
        label_col : str, default='PROVENIENZA'
            Name of the column containing class labels.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                             random_state=self.random_state)
        y = X[label_col]
        for train_measures_idx, test_measures_idx in cv.split(X, y):
            yield train_measures_idx, test_measures_idx

    def suite_name(self):
        return 'all_measures'

    def __repr__(self):
        args = {}
        if self.n_splits != 3:
            args['n_splits'] = self.n_splits
        if self.shuffle != False:
            args['shuffle'] = self.shuffle
        if self.random_state is not None:
            args['random_state'] = self.random_state
        
        arg_string = ', '.join([f'{name}={value}'
                                for name, value in args.items()])
        return f'MeasureStratifiedKFold({arg_string})'

    def __str__(self):
        return self.__repr__()


class FragmentStratifiedKFold:
    """K-Folds cross-validator with stratification on fragments.

    Provides train/test indices to split data in train/test sets.
    Stratification is done on labels for all measures, with grouping on
    fragments: folds preserve the fraction of samples for class, though
    grouping togheter in a same fold all measures of a same fragment.
    
    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    num_samples : int or None, default=None
        Number of measures to be sampled for each fragment, or None if
        all measures should be retrieved.
    stratify_col : str, default 'FRAMMENTO'
        Name of the column containing fragments.
    label_col : str, default 'PROVENIENZA'
        Name of the column containing class labels.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tarquinia.model_selection import FragmentStratifiedKFold
    >>> X = pd.DataFrame({'FRAMMENTO': [0, 0, 1, 1, 2, 2, 2, 3, 3, 3],
    ...                   'feature': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ...                   'PROVENIENZA': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})
    >>> fskf = FragmentStratifiedKFold(n_splits=2)
    >>> print(fskf)
    FragmentStratifiedKFold(n_splits=2)
    >>> for train_index, test_index in fskf.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    >>>     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    TRAIN: [2 3 7 8 9] TEST: [0 1 4 5 6]
    TRAIN: [0 1 4 5 6] TEST: [2 3 7 8 9]
    
    Notes
    -----
    The implementation expects to work with datasets encoded as pandas
    dataframes, containing two attributes describing class labels and fragments
    (see documentation of the `split` method). Using other data structures such
    as numpy arrays is not allowed.
    """

    def __init__(self, n_splits=3, num_samples=None,
                 stratify_col='FRAMMENTO', label_col='PROVENIENZA',
                 shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError(f'number of folds should be greater than 1 ('
                             f'provided n_splits={n_splits})')
        self.n_splits = n_splits
        self.num_samples = num_samples
        self.stratify_col = stratify_col
        self.label_col = label_col
        self.shuffle = shuffle
        self.random_state = random_state

    def _get_cases(self, X, fragment):
        """Return indices in a dataset corresponding to a given fragment.

        Provides the indices of cases in a dataset which refer to a fixed
        fragment, either sampling a same number of cases or considering all
        possible results.
    
        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Number of folds. Must be at least 2.
        fragment : int
            Number describing a fragment.

        Yields
        ------
        indices : ndarray of int
            Indices in X of cases having fragment equal to `fragment`.
        
        Notes
        -----
        When `num_samples` is assigned an integer value, a sample of
        `num_samples` cases referring to the specified fragment is drawn;
        when `num_samples` is `None` all cases referring to the fragment
        are considered.
        """

        selected = X.loc[X[self.stratify_col]==fragment]
        if self.num_samples is not None:
            selected = selected.sample(self.num_samples)
        return selected.index.values

    def _get_measures_idx(self, idx, fragments, fragments_idx):
        """Return indices in a dataset corresponding to a set of fragments.

        Provides the indices of cases in a dataset which refer to a set of
        fragments, either sampling a same number of cases or considering all
        possible results.
    
        Parameters
        ----------
        idx : pandas dataframe of shape (n_samples, n_features)
            Number of folds. Must be at least 2.
        fragments : ndarray of int
            Set of fragments to be considered.

        Yields
        ------
        indices : ndarray of int
            Indices in X of selected cases referring to the specfied fragments.
        
        Notes
        -----
        When `num_samples` is assigned an integer value, a sample of
        `num_samples` cases referring to each fragment is drawn; when
        `num_samples` is `None` all cases referring to the selected fragments
        are considered.
        """

        selection = [self._get_cases(idx, fragments[i])
                     for i in fragments_idx]
        return np.concatenate(selection)

    def _get_fragments(self, X):
        """Return the unique identifiers of fragments in a dataset.

        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Dataset to be considered.

        Yields
        ------
        indices : ndarray of int
            Different values fragment identifiers in X.
        """
        return X[self.stratify_col].unique()

    def split(self, X):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : pandas dataframe of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            Note that, unlike sklearn, the dataframe should include two columns
            describing fragments and labels, respectively.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """

        idx = X.reset_index().loc[:, [self.stratify_col]]
        y = X.reset_index()[self.label_col]

        fragments = self._get_fragments(X)
        fragments_index = [idx[idx[self.stratify_col]==f].iloc[0].name
                          for f in fragments]
        fragments_label = y[fragments_index]
        assert(len(fragments) == len(fragments_label))

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                             random_state=self.random_state)

        for train_fragments_idx, test_fragments_idx in \
                                    cv.split(fragments, fragments_label):
            train_measures_idx = self._get_measures_idx(idx, fragments,
                                                       train_fragments_idx)

            test_measures_idx = self._get_measures_idx(idx, fragments,
                                                      test_fragments_idx)

            assert(not np.intersect1d(
                X.iloc[train_measures_idx][self.stratify_col].unique(),
                X.iloc[test_measures_idx][self.stratify_col].unique()))

            yield train_measures_idx, test_measures_idx

    def suite_name(self):
        return 'frag_nosample' if self.num_samples is None \
                               else f'frag_sample-{self.num_samples}'

    
    def __repr__(self):
        args = {}
        if self.n_splits != 3:
            args['n_splits'] = self.n_splits
        if self.shuffle != False:
            args['shuffle'] = self.shuffle
        if self.random_state is not None:
            args['random_state'] = self.random_state
        
        arg_string = ', '.join([f'{name}={value}'
                                for name, value in args.items()])
        return f'FragmentStratifiedKFold({arg_string})'
    
    def __str__(self):
        return self.__repr__()


def kfold_factory(cv_class, num_splits, num_samples):
    """Factory function to create a KFold object.

    Parameters
    ----------
    cv_class : class
        Class of the cross-validation object to be created.
    num_splits : int
        Number of folds. Must be at least 2.
    num_folds : int
        Number of folds. Must be at least 2.

    Returns
    -------
    cv : cv_class
        Cross-validation object.
    """

    if cv_class is FragmentStratifiedKFold:
        return cv_class(n_splits=num_splits, num_samples=num_samples)
    elif cv_class is MeasureStratifiedKFold:
        return cv_class(n_splits=num_splits)
    else:
        raise ValueError(f'cv_class {cv_class} not supported')
