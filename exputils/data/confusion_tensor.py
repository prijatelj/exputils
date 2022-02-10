"""Confusion tensor generalizaiton of the confusion matrix."""
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


class OrderedConfusionTensor(object):
    """A Tensor of KxCxC where K axisension represents the ordered confusion,
    and max(K) == C with each individual CxC along the K axisenison is the
    ordered slice of the ordered predictions. Looking at one slice of the
    internal confusion tensor is NOT the top-k confusion matrix, but may be
    used to obtain the top-k measures.
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
        labels : = None
        top_k : = None
        """
        # NominalDataEncoder of labels
        self.label_enc = NDE(labels)

        # TODO generalize about axis, esp in the k loop of unique indices
        #axis = 1

        assert(preds.shape[1] == len(labels))

        n_classes = len(self.label_enc)

        if top_k is None:
            top_k = n_classes
        else:
            assert(isinstance(top_k, int))

        # Get top-k predictions, assuming prediction
        top_preds = np.argsort(preds, axis=1)[:, ::-1][:, :top_k]

        # Cast both targets and preds to their args for a confusion matrix
        # based on label_enc
        if not targets_idx:
            targets = self.label_enc.encode(targets).reshape(targets.shape)

        # TODO decide if supporting weights is even necessary as this is counts
        #    if weights is not None:
        #        unique_idx_n *= weights

        # TODO Repeat this top_k times
        ordered_cms = np.zeros([top_k, n_classes, n_classes])
        for k in range(top_k):
            unique_idx, unique_idx_counts = np.unique(
                np.stack([targets, top_preds[:, [k]]], axis=0),
                return_counts=True,
                axis=1,
            )
            np.add.at(
                ordered_cms[k],
                tuple(unique_idx.T.reshape(-1, unique_idx.T.shape[-1]).T),
                unique_idx_counts.ravel(),
            )
        self.tensor = ordered_cms

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
        return

    def accuracy(self, k=None):
        """Top-k accuracy. K is maxed by default if not given. k inclusive"""
        assert(k is None or k > 0)
        if k is None:
            return (
                self.tensor.diagonal(axis1=1, axis2=2).sum()
                / self.tensor[0].sum()
            )
        return (
            self.tensor[:k].diagonal(axis1=1, axis2=2).sum()
            / self.tensor[0].sum()
        )

    # TODO consider how top-k may relate to the discrete mutual information.
    # research and look up if any existing measures for this.

def get_cm_tensor(targets, top_preds, top_k, n_classes, axis=0):
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
