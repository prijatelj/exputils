"""Unofficial test. Just a quick ipython script test."""
from exputils.data.confusion_tensor import OrderedConfusionTensor
import numpy as np

pred_top_k = np.array([[1, 0, 2],
       [1, 2, 0],
       [2, 0, 1],
       [0, 1, 2],
       [1, 2, 0]])

truth = np.array([1, 2, 0, 2, 0]).reshape(-1,1)

# I know this function works from before
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
            #unique_idx.T,
            tuple(unique_idx.T.reshape(-1, unique_idx.T.shape[-1]).T),
            unique_idx_counts.ravel(),
        )
    return ordered_cms

from scipy.stats import dirichlet
diri = dirichlet([1/3]*3)
samples = diri.rvs(5, random_state=0)

ct = OrderedConfusionTensor(truth, samples, [0,1,2])

top_cms = get_cm_tensor(truth, pred_top_k, 3, 3)

(top_cms == ct.tensor).all()

# Check NDE label encoder when class labels are str
ct_str = OrderedConfusionTensor(
    truth.astype(str),
    samples,
    ['0','1','2'],
    targets_idx=False,
)

(top_cms == ct_str.tensor).all()
