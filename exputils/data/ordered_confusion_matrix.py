"""Ordered confusion matricies for calculating top-k measures."""
import os

import numpy as np
ma = np.ma

from exputils.data.labels import NominalDataEncoder as NDE


def scatter_nd_numpy(indices, updates, shape, target=None):
    """Equivalent to tf.scatter_nd()."""
    # TODO match tf scatter_nd interface
    # This does not work for recurring indices, which is necessary for CMs
    if target is None:
        target = np.zeros(shape, dtype=updates.dtype)
    np.add.at(
        target,
        tuple(indices.reshape(-1, indices.shape[-1]).T),
        updates.ravel(),
    )
    if target is None:
        return target


class OrderedConfusionMatrices(object):
    """A Tensor of KxCxC where K dimension represents the ordered confusion,
    and max(K) == C with each individual CxC matrix along the K dimenison is
    the ordered slice of the ordered predictions. Looking at one slice of the
    internal confusion tensor is NOT the top-k confusion matrix, but rather the
    confusion matrix when looking at n-th highest probable class. This may be
    used to obtain the top-k measures.

    Attributes
    ----------
    tensor : np.array
    label_enc : NominalDataEncoder

    Notes
    -----
    The ordered k confusion matrices loses the sample informaiton where the
    sample is the probability vector for the predicted class, which contains
    the order and relation of predicted classes for that sample. This only
    contains the confusion matrix for the k-th confusion matrix where the first
    k=1 confusion matrix is the standad confusion matrix (top-1). The second
    confusion matrix looks at the second most probable class in the predicted
    probability vectors, and so on for the values of k.
    """
    def __init__(
        self,
        targets,
        preds,
        labels,
        top_k=None,
        #axis=1,
        sort_labels=False,
        targets_idx=True,
        #weights=None,
    ):
        """Calculates the top-k confusion matrix of occurrence from the pairs.

        Args
        ----
        targets : array-like
            Vector of labels with length N.
        preds : array-like
            shape [N, C] for C classes, assumed to match labels
        labels : list = None
        top_k : int = None
        targets_idx : bool = True
            If True (default), casts the targets to their index encoding.
            The calculation of the top k-th confusion matrix is dependent upon
            the use of integer index encoding representaiton of the symbols.
        """
        # NominalDataEncoder of labels to track symbol to the index
        self.label_enc = NDE(labels)

        # TODO generalize about axis, esp in the k loop of unique indices
        #axis = 1

        assert(preds.shape[1] == len(labels))

        n_classes = len(self.label_enc)

        if top_k is None:
            top_k = 1
        else:
            assert(isinstance(top_k, int))

        # Get top-k predictions, assuming prediction
        top_preds = np.argsort(preds, axis=1)[:, ::-1][:, :top_k]

        # Cast both targets and preds to their args for a confusion matrix
        # based on label_enc
        if not targets_idx:
            targets = self.label_enc.encode(targets)

        # TODO decide if supporting weights is even necessary as this is counts
        #    if weights is not None:
        #        unique_idx_n *= weights
        self.tensor = get_cm_tensor(
            targets,
            top_preds,
            top_k,
            n_classes,
            axis=0,
        )

    def __add__(self, other):
        """Necessary for checking the changes over increments."""
        raise NotImplementedError()

    # TODO obtain BinaryTopKConfusionMatrix / Tensor and its associated
    # measures.
    def get_per_class_binary_top(self, k, axis1=1, axis2=2):
        """Returns a matrix of [C, 3] for C classes and the outcome per class
        using top-k logic of correct prediction (true positive per class)
        versus the sum of predicting incorrectly (false positive per class).
        The third vector is the false negatives.
        """
        true_pos = self.tensor.diagonal(axis1=1, axis2=2).sum(0)
        non_diag = ma.masked_array(
            self.tensor,
            mask=np.stack([np.eye(k, dtype=bool)] * k),
        )
        false_pos = non_diag.sum(1)
        false_neg = non_diag.sum(2)
        #true_positives
        return np.hstack((true_pos, false_pos, false_neg))

    def accuracy(self, k=None):
        """Top-k accuracy. K is maxed by default if not given. k inclusive"""
        assert(k is None or isinstance(k, int))
        if k is None or k == 1:
            return (
                self.tensor.diagonal(axis1=1, axis2=2).sum()
                / self.tensor[0].sum()
            )
        elif k < 1:
            raise ValueError('k is to be >= 1 or None, but given k < 1!')
        return (
            self.tensor[:k].diagonal(axis1=1, axis2=2).sum()
            / self.tensor[0].sum()
        )

    # TODO consider how top-k may relate to the discrete mutual information.
    # research and look up if any existing measures for this.
    def save(
        self,
        filepath,
        conf_mat_key='ordered_confusion_matrices',
        overwrite=False,
        *args,
        **kwargs,
    ):
        """Saves the current confusion matrix to the given filepath."""
        if not isinstance(filetype, str):
            raise NotImplementedError(
                'Only str filepaths are supported for saving.',
            )

        ext = os.path.splitext(filepath)[-1]

        if ext not in {'.hdf5', '.h5'}:
            raise TypeError(
                f'Expected file extention: ".hdf5", or ".h5"; not `{ext}`',
            )

        filepath = create_filepath(filepath, overwrite=overwrite)

        with h5py.File(filepath, 'r') as h5f:
            h5f['labels'] = self.labels.astype(np.string_)
            h5f[conf_mat_key] = self.tensor

    @staticmethod
    def load(
        self,
        filepath,
        sep=',',
        filetype=None,
        names=None,
        conf_mat_key='ordered_confusion_matrices',
        *args,
        **kwargs,
    ):
        """Convenience function that loads the confusion matrix from the given
        filepath in given common formats. Loading from CSV relies on
        pandas.read_csv(*args, **kwargs).
        """
        raise NotImplementedError()

        if filetype is None:
            # Infer filetype from filepath if filetype is not given
            parts = filepath.rpartition('.')

            if not parts[0]:
                raise ValueError(' '.join([
                    'filetype is `None` and no file extention present in',
                    'given filepath.',
                ]))

            if parts[2] and os.path.sep in parts[2]:
                raise ValueError(' '.join([
                    'filetype is `None` and no file extention present in',
                    'given filepath.',
                ]))

            filetype = parts[2].lower()
            if filetype == 'csv':
                sep = ','
                #filetype = 'csv'
            elif filetype == 'tsv':
                sep = '\t'
                #filetype = 'tsv'
            elif filetype == 'hdf5' or filetype == 'h5':
                filetype = 'hdf5'
            else:
                raise ValueError(' '.join([
                    'filetype is `None` and file extention present in',
                    'given filepath is not "csv", "tsv", "hdf5", or "h5".',
                ]))
        elif isinstance(filetype, str):
            # lowercase filetype and check if an expected extension
            filetype = filetype.lower()
            if filetype not in {'csv', 'tsv', 'hdf5', 'h5'}:
                raise TypeError(' '.join([
                    'Expected filetype to be a str: "csv", "tsv", "hdf5", or',
                    f'"h5", not `{filetype}`',
                ]))
        else:
            raise TypeError(' '.join([
                'Expected filetype to be a str: "csv", "tsv", "hdf5", or',
                f'"h5", not of type `{type(filetype)}`',
            ]))

        if filepath == 'csv' or 'tsv':
            loaded_conf_mat = pd.read_csv(
                filepath,
                sep=sep,
                names=names,
                *args,
                **kwargs,
            )

            assert (
                len(loaded_conf_mat.shape) == 2
                and loaded_conf_mat.shape[0] == loaded_conf_mat.shape[1]
            )

            if names is None:
                labels = np.array(loaded_conf_mat.columns)
            else:
                labels = None
        else:  # HDF5
            with h5py.File(filepath, 'r') as h5f:
                if 'labels' in h5f.keys():
                    if names is not None:
                        logging.warning(' '.join([
                            '`names` is provided while "labels" exists in the',
                            'hdf5 file! `names` is prioritized of the labels',
                            'in hdf5 file.',
                        ]))
                        labels = names
                    else:
                        labels = h5f['labels'][:]

                loaded_conf_mat = h5f[conf_mat_key][:]

        return ConfusionMatrix(loaded_conf_mat, labels=labels)


def get_cm_tensor(targets, top_preds, top_k, n_classes, axis=0):
    # TODO make sparse numpy matrix/tensor w/ zero as fill.
    ordered_cms = np.zeros([top_k, n_classes, n_classes])
    for k in range(top_k):
        unique_idx, unique_idx_counts = np.unique(
            np.stack([targets, top_preds[:, [k]]], axis=axis),
            return_counts=True,
            axis=1,
        )
        np.add.at(
            ordered_cms[k],
            tuple(unique_idx.T.reshape(-1, unique_idx.T.shape[-1]).T),
            unique_idx_counts.ravel(),
        )
    return ordered_cms
