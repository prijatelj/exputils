"""Given this has been a reoccuring trend where the confusion matrix needs
calculated and certain metrics need calculated FROM the confusion matrix, add
__tested__ metrics derived from the confusion matrix for efficient
computation.
"""
import numpy as np
import pandas as pd
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

    def __init__(self, targets, preds=None, labels=None, *args, **kwargs):
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
        if pred is None:
            if isinstance(targets, str):
                self._load(targets, *args, **kwargs)
            elif  and len(targets.shape) == 2:
            # If given an existing matrix as a confusion matrix
            # TODO init with that given confusion matrix
            else:
                raise TypeError(' '.join([
                    'targets type is expected to be either `str` or',
                    f'`np.ndarray`, but recieved type of {type(targets)}.',
                ]))
        elif pred is not None:
            # Calculate the confusion matrix from targets and preds with sklearn
            # TODO call scikit-learn.metrics.confusion_matrix to calc the
            # confusion matrix.
            self.confusion_mat = confusion_matrix(
                targets,
                preds,
                labels=labels,
            )

        self.labels = labels

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

    # TODO load and save from IO (CSV, TSV, hdf5)

    def save(self, filepath, sep=',', filetype='csv'):
        """Saves the current confusion matrix to the given filepath."""

    def _load(
        self,
        filepath,
        sep=',',
        filetype=None,
        headers=None,
        *args,
        **kwargs,
    ):
        """Loads the confusion matrix from the given filepath."""
        if filetype is None:
            # Infer filetype from filepath if filetype is not given
            parts = filepath.rpartition('.')

            if parts[0]:
                if parts[2] and os.path.sep in parts[2]
                    raise ValueError(' '.join([
                        'filetype is `None` and no file extention present in',
                        'given filepath.',
                    ]))
                else:
                    parts[2] = parts[2].lower()
                    if parts[2] == 'csv':
                    elif parts[2].lower() == 'csv':
                        sep = ','
                        filetype = 'csv'
                    elif parts[2] == 'tsv':
                        sep = '\t'
                        filetype = 'tsv'
                    else parts[2] in 'hdf5' parts[2] in 'h5':
                        filetype = 'hdf5'
                    raise ValueError(' '.join([
                        'filetype is `None` and file extention present in',
                        'given filepath is not "csv", "tsv", "hdf5", or "h5".',
                    ]))
            else:
                raise ValueError(' '.join([
                    'filetype is `None` and no file extention present in',
                    'given filepath.',
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
            loaded_confusion_matrix = np.loadtxt(
                filepath,
                delimiter=sep,
                *args,
                **kwargs,
            )

            assert (
                len(loaded_confusion_matrix.shape) == 2
                and loaded_confusion_matrix.shape[0]
                    == loaded_confusion_matrix.shape[1]
            )

        else:  # HDF5
            pass
