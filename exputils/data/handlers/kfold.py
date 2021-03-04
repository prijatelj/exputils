"""Kfold Cross Validation wrappers for optionally inverting the role of the
focus fold.
"""
from sklearn.model_selection import KFold, StratifiedKFold


def kfold_generator(
    kfolds,
    features,
    labels=None,
    shuffle=True,
    stratified=False,
    train_focus_fold=False,
    random_seed=None,
):
    """Convenience function that generates the K different indices.

    Parameters
    ----------
    kfolds : int
        The number (K) of folds to create from the data.
    features : array-like, shape (n_samples,)
        The data to be split with the number of samples being the first dim.
    labels : array-like, shape (n_samples,), optional
       The target variable for supervised learning problems. Stratification is
       done based on the labels. The number of samples is the first dimension
       of the features.
    shuffle : bool, optional (default=True)
        Whether to shuffle the data or not. Default is True to shuffle.
    stratified : bool, optional (default=False)
        Attempts to stratify the data if True. Default is False.
    train_focus_fold : bool, optional (default=False)
        Convenience parameter for swapping the order of the other_folds and
        focus_fold, specifically for when the `focus_fold` is to be used for
        training as in-sample data, instead of being used for the evaluation,
        which is the norm.
    random_state : int | RandomState | None, optional (default=None)
        The random state or seed to be used by `sklearn.model_selection.KFold`.
    """
    # Data index splitting
    if stratified:
        fold_indices = StratifiedKFold(kfolds, shuffle, random_seed).split(
            features,
            labels,
        )
    else:
        fold_indices = KFold(kfolds, shuffle, random_seed).split(
            features,
            labels,
        )

    for other_folds, focus_fold in fold_indices:
        # Set the correct train and test indices
        if train_focus_fold:
            yield focus_fold, other_folds
        else:
            yield other_folds, focus_fold


def get_kfold_idx(
    focus_fold,
    kfolds,
    features,
    labels=None,
    shuffle=True,
    stratified=False,
    train_focus_fold=False,
    random_seed=None,
):
    """Gets a specific set of indices from a setup of kfold cv."""
    for i, (train_idx, test_idx) in enumerate(kfold_generator(
        kfolds,
        features,
        labels,
        shuffle,
        stratified,
        train_focus_fold,
        random_seed,
    )):
        if i + 1 == focus_fold:
            return train_idx, test_idx
