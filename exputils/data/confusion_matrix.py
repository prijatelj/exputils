"""Given this has been a reoccuring trend where the confusion matrix needs
calculated and certain metrics need calculated FROM the confusion matrix, add
__tested__ metrics derived from the confusion matrix for efficient
computation.
"""
from sklearn.metrics import confusion_matrix

# TODO obtain confusion matrix from sklearn

# TODO obtain acc, f1, MCC, informedness, etc. from a confusion matrix

# TODO obtain acc, f1, MCC, informedness, etc. from a confusion matrix

# TODO create a confusion matrix class that manages all typical things to be
# done with the confusion matrix and bundles the parts together nicely.

class ConfusionMatrix(object):
    """Confusion matrix for nominal data that wraps the
    sklearn.metrics.confusion_matrix.
    """

    def __init__(self, targets, preds=None, labels=None):
        """
        Parameters
        ----------
        targets : np.ndarray()
            The confusion matrix to wrap, or the target labels.
        preds : np.ndarray, optional
        labels : list, optional
            The labels for the row and columns of the confusion matrix
        """
        # TODO calc the
        if pred is None and len(target.shape) == 2:
            # If given an existing matrix as a confusion matrix
            # TODO init with that given confusion matrix
        elif pred is not None:
            # TODO call scikit-learn.metrics.confusion_matrix to calc the
            # confusion matrix.

    # TODO properties/methods for the metrics able to be derived from the
    # confusion matrix
    # accuracy
    # error rate
    # precision
    # recall
    # f1
    # informedness
    # mathew's correlation coefficient (MCC)
    # ROC
    # ROC AUC
    # TOC
    # TOC AUC

    # TODO Reduction of classes including two class subsets to obtain
    # Binarization
    #   e.g. given set of labels as known and set of labels as unknown return
    #   binary confusion matrix of knowns vs uknowns

    # TODO combination of different confusion matrices objects into one.

    # TODO Visualizations
    # Confusion matrix visualized as a heat map
    # ROC
    # TOC

    # TODO load and save from IO (CSV, TSV)
