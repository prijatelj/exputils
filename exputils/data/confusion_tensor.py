"""Confusion tensor generalizaiton of the confusion matrix."""
import numpy as np

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


class OrderedConfusionTensor(ConfusionMatrix):
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
        axis=1,
        sort_labels=False,
        targets_idx=True,
        weights=None,
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

        assert(preds.shape[axis] == len(labels))

        n_classes = len(self.label_enc)

        if top_k is None:
            top_k = n_classes
        else:
            assert(isinstance(top_k, int))

        # Get top-k predictions, assuming prediction
        top_preds = np.argsort(preds, axis)[:, ::-1][:, :top_k]

        # TODO Create a confusion tensor of ordered confusion? The order is
        # informtive, and a confusion matrix/tensor or similar construct may
        # capture this information.

        # Cast both targets and preds to their args for a confusion matrix
        # based on label_enc
        if not targets_idx:
            targets = self.label_enc.encode(targets)

        # TODO Repeat targets to align w/ top_preds with the K index, then
        # scatter_nd would perform the desired k-th ordered confusion matrix
        # creation.

        # TODO decide if supporting weights is even necessary as this is counts
        #    if weights is not None:
        #        unique_idx_n *= weights

        # TODO Repeat this top_k times
        ordered_cms = np.zeros([top_k, n_classes, n_classes])
        for k in range(top_k):
            unique_idx, unique_idx_counts = np.unique(
                np.stack([targets, top_preds[:, [k]]], axis=axis),
                return_counts=True,
            )
            np.add.at(
                ordered_cms[k],
                unique_idx.T,
                #tuple(indices.reshape(-1, indices.shape[-1]).T),
                unique_idx_counts.ravel(),
            )
        self.tensor = ordered_cms

    # TODO obtain BinaryTopKConfusionMatrix / Tensor and its associated
    # measures.
