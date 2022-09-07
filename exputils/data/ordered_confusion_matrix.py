"""Ordered confusion matricies for calculating top-k measures."""
from copy import deepcopy
import os

from bidict import OrderedBidict
import h5py
import numpy as np
ma = np.ma

from exputils.data.labels import NominalDataEncoder as NDE
from exputils.data import ConfusionMatrix
from exputils.io import create_filepath

import logging
logger = logging.getLogger(__name__)


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
        preds=None,
        labels=None,
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
            If True (default), the labels are in their index encoding. Else
            (False) casts the targets to their index encoding.  The calculation
            of the top k-th confusion matrix is dependent upon the use of
            integer index encoding representaiton of the symbols.
        """
        # TODO generalize about axis, esp in the k loop of unique indices
        #axis = 1

        # NominalDataEncoder of labels to track symbol to the index
        if preds is None:
            if (
                isinstance(targets, np.ndarray)
                and len(targets.shape) == 3
                and targets.shape[1] == targets.shape[2]
            ):
                self.tensor = targets
                if top_k is not None:
                    raise ValueError(
                        'top_k given when targets is the ordered confusion '
                        'matrices.'
                    )

                if labels is None:
                    raise ValueError(
                        'OrderedConfusionMatrices without labels!'
                    )
                elif isinstance(labels, NDE):
                    self.label_enc = labels
                else:
                    self.label_enc = NDE(labels)
        else:
            if top_k is None:
                top_k = 1
            else:
                assert(isinstance(top_k, int))

            if labels is None:
                self.label_enc = NDE(list(set(targets) | set(preds)))
            elif isinstance(labels, NDE):
                self.label_enc = labels
            else:
                self.label_enc = NDE(labels)

            n_classes = len(self.label_enc)
            assert(preds.shape[1] == n_classes)

            # Get top-k predictions, assuming prediction is of shape [N, C]
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

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, val in self.__dict__.items():
            setattr(result, key, deepcopy(val, memo))
        return result

    # TODO consider nominal data enc set operators that ensure order of
    # left op right, OrderedDict / OrderedBidict cover this?
    def __add__(self, other):
        """Add two OrderedConfusionMatrices together if of the same shape w/
        same label set."""
        return self.join(other, 'left', False)

    @property
    def labels(self):
        """Returns labels as a numpy array."""
        return np.array(self.label_enc.encoder)

    @property
    def top_k(self):
        return self.tensor.shape[0]

    def join(self, other, method='left', inplace=False):
        """Joins this OrderedConfusionMatricies with another using a set method
        over their labels.

        Args
        ----
        other : OrderedConfusionMatrices
        method : str = 'left'
            Left-preferred  union only atm.
            May be a str identifier of {'union', 'left', 'right', 'intersect',
            'disjoint'}.
        inplace : bool = False
            If True, updates this self instance with the result of joining.

        Returns
        -------
        ConfusionMatrix | None
            The result of joining the two ConfusionMatrix objects if inplace is
            False, otherwise None.
        """
        if not isinstance(other, OrderedConfusionMatrices):
            raise TypeError(
                'Operator add only supported between two '
                'OrderedConfusionMatrices.'
            )
        if self.tensor.shape[0] != other.tensor.shape[0]:
            raise ValueError(
                "The OrderedConfusionMatrices' tensors dim 0 are not the "
                'same. Expects same number of top confusion matrices.'
            )
        set_self = set(self.label_enc)
        set_other = set(other.label_enc)

        intersect = list(set_self & set_other)
        self_intersect = self.label_enc.encode(intersect)
        other_intersect = other.label_enc.encode(intersect)

        #self_disjoint = self.label_enc.encode(list(set_self - set_other))
        other_disjoint = np.sort(other.label_enc.encode(
            list(set_other - set_self)
        ))

        if inplace:
            label_enc = self.label_enc
            tensor = self.tensor
        else:
            label_enc = deepcopy(self.label_enc)
            tensor = self.tensor.copy()
        n_self = len(label_enc)
        n_other = len(other_disjoint)
        n_total = n_self + n_other

        # Sum the mats together aligned by their shared labels.
        # First, add across the intersecting rows
        self_intersect_args = self_intersect.reshape(-1, 1).repeat(
            self_intersect.shape[0],
            axis=1,
        )
        tensor[:, self_intersect_args, self_intersect_args.T] += \
            other.tensor[:, other_intersect][:, :, other_intersect]

        # Add Zeros placeholders for other's disjoint labels
        new_tensor = np.zeros([tensor.shape[0], n_total, n_total])

        # Update with their values:
        # Self's confusion matrices
        new_tensor[:, :n_self, :n_self] = tensor
        if n_other > 0:
            # Other's disjoint x disjoint
            new_tensor[:, -n_other:, -n_other:] = \
                other.tensor[:, other_disjoint][:, :, other_disjoint]
            # Other's disjoint x intersect; and intersect x disjoint
            new_tensor[:, -n_other:, self_intersect] = \
                other.tensor[:, other_disjoint][:, :, other_intersect]
            new_tensor[:, self_intersect, -n_other:] = \
                other.tensor[:, other_intersect][:, :, other_disjoint]

        # Add other's disjoint labels to self label encoder
        label_enc.append(other.label_enc.decode(other_disjoint))

        if not inplace:
            # TODO use copy rather than new init.
            return OrderedConfusionMatrices(new_tensor, labels=label_enc)
        else:
            self.tensor = new_tensor

    def get_conf_mat(self):
        """Returns the top-1 ConfusionMatrix, the first matrix in tensor."""
        return ConfusionMatrix(self.tensor[0], labels=self.label_enc)

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
        # TODO unit test this.
        assert(k is None or isinstance(k, int))
        if k is None or k == 1:
            return (
                self.tensor[0].diagonal().sum()
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
        if not isinstance(filepath, str):
            raise NotImplementedError(
                'Only str filepaths are supported for saving.',
            )

        ext = os.path.splitext(filepath)[-1]

        if ext not in {'.hdf5', '.h5'}:
            raise TypeError(
                f'Expected file extention: ".hdf5", or ".h5"; not `{ext}`',
            )

        filepath = create_filepath(filepath, overwrite=overwrite)

        with h5py.File(filepath, 'w') as h5f:
            h5f['labels'] = np.array(self.label_enc).astype(np.string_)
            h5f[conf_mat_key] = self.tensor

    @staticmethod
    def load(
        filepath,
        labels=None,
        conf_mat_key='ordered_confusion_matrices',
    ):
        """Convenience function that loads the confusion matrix from the given
        filepath in given common formats. Loading from CSV relies on
        pandas.read_csv(*args, **kwargs).
        """
        ext = os.path.splitext(filepath)[-1]

        if ext not in {'.hdf5', '.h5'}:
            raise TypeError(
                f'Expected file extention: ".hdf5", or ".h5"; not `{ext}`',
            )

        with h5py.File(filepath, 'r') as h5f:
            if 'labels' in h5f.keys():
                if labels is not None:
                    logger.warning(
                        '`names` is provided while "labels" exists in the '
                        'hdf5 file! `names` is prioritized of the labels '
                        'in hdf5 file.'
                    )
                else:
                    labels = [label.decode() for label in h5f['labels'][:]]

            loaded_conf_mat = h5f[conf_mat_key][:]

        return OrderedConfusionMatrices(loaded_conf_mat, labels=labels)


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
