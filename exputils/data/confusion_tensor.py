"""Confusion tensor generalizaiton of the confusion matrix."""
import numpy as np

from exputils.data.labels import NominalDataEncoder as NDE

class OrderedConfusionTensor(ConfusionMatrix):
    """A Tensor of KxCxC where K dimension represents the ordered confusion,
    and max(K) == C with each individual CxC along the K dimenison is the
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
        dim=1,
        sort_labels=False,
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

        assert(pred.shape[dim] == len(labels))

        if top_k is None:
            top_k = len(self.label_enc)
        else:
            assert(isinstance(top_k, int))

        # Get top-k predictions, assuming prediction
        top_preds = np.argsort(preds, dim)[:, ::-1][:, :top_k]

        # TODO Boolean check if target in top_preds: Only good for binary CM

        # TODO Create a confusion tensor of ordered confusion? The order is
        # informtive, and a confusion matrix/tensor or similar construct may
        # capture this information.

        # TODO cast both targets and preds to their args for a confusion matrix
        # based on label_enc


        self.tensor = ...
