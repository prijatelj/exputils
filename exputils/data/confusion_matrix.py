"""Given this has been a reoccuring trend where the confusion matrix needs
calculated and certain metrics need calculated FROM the confusion matrix, add
__tested__ metrics derived from the confusion matrix for efficient
computation.
"""
import logging
import os

import h5py
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from scipy.stats import gmean, entropy
from sklearn.metrics import confusion_matrix

from exputils.io import create_filepath

# TODO ConfusionTensor: generalize confusion matrix to multiple discrete RVs
# with a ConfusionTensor class, which would be the parent to ConfusionMatrix

# TODO consider ways to interlink this with LabelEncoder better. If necessary

class ConfusionMatrix(object):
    """Confusion matrix for nominal data that wraps the
    sklearn.metrics.confusion_matrix. Rows are the known labels and columns are
    the predictions.
    """
    def __init__(self, targets, preds=None, labels=None, *args, **kwargs):
        """
        Parameters
        ----------
        targets : np.ndarray()
            The confusion matrix to wrap, or the target labels.
        preds : np.ndarray, optional
        labels : np.ndarray, optional
            The labels for the row and columns of the confusion matrix
        """

        # TODO Add optional class weights attribute, allowing it to be
        # overwritten for when the measures are claculated via a param

        # TODO add sample weights param to init here to affect calc of conf

        # TODO perhaps add a matrix of belief/reliability per pairwise class
        # expressed by some probability per element

        if preds is None:
            if (
                (
                    isinstance(targets, np.ndarray)
                    or isinstance(targets, pd.DataFrame)
                )
                and len(targets.shape) == 2
                and targets.shape[0] == targets.shape[1]
            ):
                # If given an existing matrix as a confusion matrix
                self.mat = np.array(targets)

                if isinstance(labels, list):
                    self.labels = np.array(labels)
                elif isinstance(labels, np.ndarray):
                    self.labels = labels
                elif isinstance(targets, pd.DataFrame):
                    self.labels = np.array(targets.columns)
                else:
                    self.labels = np.arange(len(self.mat))
            else:
                raise TypeError(' '.join([
                    'targets type is expected to be of type `np.ndarray`, but',
                    f'recieved type of {type(targets)}.',
                ]))
        elif preds is not None:
            # Calculate the confusion matrix from targets and preds with sklearn
            if isinstance(labels, list):
                self.labels = np.array(labels)
            elif isinstance(labels, np.ndarray):
                self.labels = labels
            else:
                self.labels = np.array(list(set(targets) | set(preds)))

            self.mat = confusion_matrix(
                targets,
                preds,
                labels=self.labels,
                *args,
                **kwargs,
            )

    def __add__(self, other):
        """Add two ConfusionMatrices together if of the same shape w/ same
        label set."""
        if not isinstance(other, ConfusionMatrix):
            raise TypeError(
                'Operator add only supported between two ConfusionMatrices.'
            )

        if len(self.labels) != len(other.labels):
            raise ValueError(' '.join([
                'The two ConfusionMatrices do not have the same number of',
                'labels!',
            ]))
        if not self.labels == other.labels:
            if not set(self.labels) == set(other.labels):
                raise ValueError(
                    'The other ConfusionMatrix does not have the same labels!'
                )

            # TODO reorganize other to be added correctly to existing conf mat
            raise NotImplementedError('Same labels of different order.')

        return self.mat + other.mat

    def reduce(self, labels, reduced_label, inverse=False):
        """Reduce confusion matrix to smaller size by mapping labels to one.

        Parameters
        ----------
        labels : list | np.ndarray
            Labels to be reduced to a single label. If `inverse` is True, then
            all the labels in the confusion matrix NOT included in this list
            are to be reduced.
        reduced_label : object
            The label to replace the others.
        inverse : bool, optional
            If True, all of the labels in the confusion matrix NOT in `labels`
            are reduced instead.

        Returns
        -------
        ConfusionMatrix
            The resulting reduced confusion matrix.
        """
        if inverse:
            if reduced_label in self.labels and reduced_label in labels:
                labels = np.delete(
                    labels,
                    np.where(labels == reduced_label)[0][0],
                )
        else:
            if reduced_label in self.labels and reduced_label not in labels:
                labels = np.append(labels, reduced_label)

        # TODO the use of the nominal label encoder would be good here.
        mask = np.zeros(len(self.labels)).astype(bool)
        for label in labels:
            mask |= self.labels == label

        if inverse:
            # Numpy mask in where= ignores False from the calculation.
            not_mask = mask
            mask = np.logical_not(mask)
        else:
            not_mask = np.logical_not(mask)

        # Construct reduced st reduced label is last index
        reduced_cm = np.vstack((
            np.hstack((
                self.mat[not_mask][:, not_mask],
                self.mat[not_mask][:, mask].sum(1, keepdims=True),
            )),
            np.hstack((
                self.mat[mask][:, not_mask].sum(0, keepdims=True),
                self.mat[mask][:, mask].sum(keepdims=True),
            )),
        ))

        return ConfusionMatrix(
            reduced_cm,
            labels=np.append(self.labels[not_mask], reduced_label),
        )

    # TODO methods for the metrics able to be derived from the confusion matrix
    # f score (1 and beta)
    # informedness # This may not generalize to multi class
    # ROC
    # ROC AUC
    # TOC
    # TOC AUC

    def accuracy(self, label_weights=None):
        #return (self.true_pos + self.true_negatives) / all = Trues / all
        if label_weights is not None:
            raise NotImplementedError('Use sklearn.metrics on the samples')
        return np.diagonal(self.mat).sum() / self.mat.sum()

    def error_rate(self, label_weights=None):
        if label_weights is not None:
            raise NotImplementedError('Use sklearn.metrics on the samples')
        return 1.0 - self.accuracy()

    def true_rate(self, label=None, average=False, label_weights=None):
        """Recall, sensitivity, hit rate, or true positive rate. This is
        calculated the same as specificity, selectivity or true negative rate,
        but on the different classes.

        Parameters
        ----------
        average : bool
            if True, averages all true class rates together. Otherwise, returns
            the true rate per class in order of labels.
        """
        # TODO check this and make a unit test for it.
        #recalls = np.diagonal(self.mat) / self.mat.sum(axis=1)
        if label is None:
            tp = self.true_positive()
            return tp / (tp + self.false_negative())

        tp = self.true_positive(label)
        return tp / (tp + self.false_negative(label))

        """
        if average:
            if label_weights is not None:
                raise NotImplementedError('Use sklearn.metrics on the samples')
            # Provide the averaged true class rates (balanced accuracy)
            return recalls.mean()

        # Provide the recall per class
        return recalls
        """

    def true_positive(self, label=None):
        if label is None:
            return np.diag(self.mat)
        return np.diag(self.mat)[label]

    def false_positive(self, label=None):
        """Calculates the False Positives either of every class or a specified
        slice.

        Parameters
        ----------
        label : np.ndarry slice
            A slice of the conufsion matrix indices, NOT the labels themselves
        """
        # TODO again, support for NominalDataEncoder would make labels able to
        # support the actual labels
        if label is None:
            return self.mat.sum(0) - np.diag(self.mat)
        return self.mat[label].sum() - self.mat[label, label]

    def false_negative(self, label=None):
        """Calculates the False Negatives of every class or a specified slice.
        """
        if label is None:
            return self.mat.sum(1) - np.diag(self.mat)
        mask = np.ones(self.mat.shape[0]).astype(bool)
        mask[label] = False
        return self.mat[:, label].sum() - self.mat[label, label]

    def false_rates(self, label):
        """Calcuate the False Positive/Negative Rates for a single label."""
        tp = self.true_positive(label)
        fp = self.false_positive(label)
        fn = self.false_negative(label)
        tn = self.mat.sum() - tp - fp - fn

        logging.debug('tp = %f', tp)
        logging.debug('tn = %f', tn)
        logging.debug('fp = %f', fp)
        logging.debug('fn = %f', fn)

        fpr = fp / (fp + tn)
        fnr = fn / (tp + fn)

        return fpr, fnr

    def f_score(self, beta=1, average='macro', label_weights=None):
        # average match sklearn f1_score: None, "binary", micro, macro,
        # weighted
        raise NotImplementedError('Use sklearn.metrics on the samples')

    def mcc(self, label_weights=None):
        """Mathew's Correlation Coefficient generalized to multiple classes,
        R_k, a generalizatin of Pearson's correlation coefficient.
        """
        if label_weights is not None:
            raise NotImplementedError()

        actual = self.mat.sum(1)
        predicted = self.mat.sum(0)
        correct_pred = np.diagonal(self.mat).sum()
        total_sqrd = self.mat.sum()**2

        # TODO be aware that NaNs occur at times, and the below may need a
        # nonzero mask like mutual information did!

        return (
            (correct_pred * self.mat.sum() - np.dot(actual, predicted)) /
            (
                np.sqrt(total_sqrd - np.dot(predicted, predicted))
                * np.sqrt(total_sqrd - np.dot(actual, actual))
            )
        )

    def mutual_information(self, normalized=None, weights=None, base=None):
        """The confusion matrix is the joint probability mass function when
        the values are divided by the total of the confusion matrix.

        Parameters
        ----------
        normalized : str, optional
            The method of normalizing the mutual information. Options include:
            'arithmetic' mean, 'geometric' mean, 'min'imum entropy, 'max'imum
            entropy, 'add' the entropies to obtain the redundancy, 'harmonic'
            mean of the two uncertainty coefficients, 'information quality
            ratio' aka a normalized mutual information as a special case of
            dual total correlation.
        weights : np.ndarray, optional
            A 2 by 2 array to weight each element of the discrete joint
            distribution.
        base : {None, 2}
            Whether to use the natural base or base 2. Defaults to 'e' for the
            natural logarithm to match with default expectation of
            scikit-learn.metrics.mutual_info_score.
        """
        # TODO weighted variants, more normalized variants, adjusted, etc.
        # TODO replicate the normalization method in scikitlearn but from a
        # given confusion matrix.
        #   NMI using "arithemtic" in scikit learn is: MI / mean(H(X),H(Y))
        #   thus, it follow geometric mean is similarly used, as is min() and
        #   max()

        # TODO I did the check and they are close but slight difference occurs
        # at the 16th decimal place and on, which make sense cuz of float
        # errors, but I am uncertain if they have a workaround to ensure it is
        # more exact in the scikit-learn code.
        # Using test example actual = [3, 2, 0, 0, 4] and pred = [4, 2, 3, 0,
        # 1], The below MI and NMI for min, max, & arithmetic all matched ==,
        # but geometric did not match exactly. it was off by ~1e-16.
        # This may be due to not removing zero prob items from  marginals in
        # entropy calc?
        # Oh hey, the above is one unit test I can add. LOL

        if base is None:
            log = np.log
        elif base == 2:
            log = np.log2
        else:
            raise ValueError(f'Unexpected value for `base`: {base}')

        joint_distrib = self.mat / self.mat.sum()
        marginal_actual = joint_distrib.sum(1).reshape(-1,1)
        marginal_pred = joint_distrib.sum(0).reshape(1,-1)

        # TODO need a unit test for this; have confirmed matches
        # sklearn.metrics.mutual_info_score, tho, when using natural base 'e'
        denom = np.dot(marginal_actual, marginal_pred)

        # Mask nonzero joint to avoid unnecessary zero division
        nonzero = joint_distrib != 0

        if (
            isinstance(weights, np.ndarray)
            and weights.shape == joint_distrib.shape
        ):
            # Weighted MI from Guiasu 1977
            mutual_info = (
                weights.flatten() * joint_distrib[nonzero]
                * log(joint_distrib[nonzero] / denom[nonzero])
            ).sum()
        else:
            mutual_info = (
                joint_distrib[nonzero]
                * log(joint_distrib[nonzero] / denom[nonzero])
            ).sum()

        # TODO able to avoid re-calc of marginals by simply calling scipy
        # entropy here! Also avoiding two func calls.
        if normalized is None:
            return mutual_info
        if normalized == 'arithmetic':
            return mutual_info / np.mean(
                (self.entropy(0, base=base), self.entropy(1, base=base))
            )
        if normalized == 'geometric':
            return mutual_info / gmean(
                (self.entropy(0, base=base), self.entropy(1, base=base))
            )
        if normalized == 'min':
            return mutual_info / np.minimum(
                self.entropy(0, base=base),
                self.entropy(1, base=base),
            )
        if normalized == 'max':
            return mutual_info / np.maximum(
                self.entropy(0, base=base),
                self.entropy(1, base=base),
            )
        if normalized == 'add':
            # Redundancy
            return mutual_info / (
                self.entropy(0, base=base) + self.entropy(1, base=base)
            )
        if normalized == 'harmonic':
            # Symmetric uncertainty: harmonic mean of the 2 uncertainty coef.s
            return mutual_info / (2 * (
                self.entropy(0, base=base) + self.entropy(1, base=base)
            ))
        if normalized == 'information quality ratio':
            # Information Quality Ratio (IQR) or special case of dual total
            # correlation
            return mutual_info / (
                self.entropy(0, base=base)
                + self.entropy(1, base=base)
                - mutual_info
            )

        raise ValueError('Unexpected value for `normalized`: {normalized}')

    def entropy(self, axis, base=None):
        """Returns the entropy of either the predictions or actual values."""
        if axis == 0 or axis == 'pred' or axis == 'predicted':
            marginal = self.mat.sum(0) / self.mat.sum()
        elif axis == 1 or axis == 'actual':
            marginal = self.mat.sum(1) / self.mat.sum()
        else:
            raise ValueError(f'Unexpected value for `axis`: {axis}')
        return entropy(marginal, base=base)

    # TODO Reduction of classes including two class subsets to obtain
    # Binarization
    #   e.g. given set of labels as known and set of labels as unknown return
    #   binary confusion matrix of knowns vs uknowns

    # TODO combination of different confusion matrices objects into one.

    # TODO Visualizations
    # ROC
    # TOC

    def heatmap(
        self,
        filepath=None,
        overwrite=False,
        **kwargs,
    ):
        """Confusion matrix visualized as a heat map using plotly."""
        if self.labels is None:
            labels = np.arange(len(self.mat))
        else:
            labels = self.labels

        # Provide more informative (overridable) default layout for the heatmap
        layout_kwargs = dict(
            title = '<b>Confusion Matrix</b>',
            xaxis = {'side': 'top'},
            yaxis = {'autorange': 'reversed'},
        )
        # If layout override exists, then update defaults
        if 'layout' in kwargs:
            if isinstance(kwargs['layout'], dict):
                layout_kwargs.update(kwargs['layout'])
                kwargs['layout'] = go.Layout(**layout_kwargs)
            else:
                layout_kwargs.update(kwargs['layout'])
                kwargs['layout'] = go.Layout(**layout_kwargs).update(
                    **kwargs['layout'],
                )
        else:
            kwargs['layout'] = go.Layout(**layout_kwargs)


        fig = go.Figure(
            data=go.Heatmap(
                z=self.mat,
                x=labels,
                y=labels,
                **kwargs.pop('data', {}),
            ),
            **kwargs,
        )

        if filepath is None:
            return fig

        # Otherwise save plot to filepath
        fig.write_image(create_filepath(filepath, overwrite=overwrite))

    def save(
        self,
        filepath,
        filetype='csv',
        conf_mat_key='confusion_matrix',
        overwrite=False,
        *args,
        **kwargs,
    ):
        """Saves the current confusion matrix to the given filepath."""
        if not isinstance(filetype, str):
            raise TypeError(' '.join([
                'Expected filetype to be a str: "csv", "tsv", "hdf5", or',
                f'"h5"; not type `{type(filetype)}`',
            ]))

        filetype = filetype.lower()
        filepath = create_filepath(filepath, overwrite=overwrite)

        if filetype == 'csv' or filetype == 'tsv':
            pd.DataFrame(self.mat, columns=self.labels).to_csv(
                filepath,
                index=False,
                *args,
                **kwargs,
            )
        else: #HDF5
            with h5py.File(filepath, 'r') as h5f:
                h5f['labels'] = self.labels.astype(np.string_)
                h5f[conf_mat_key] = self.mat

    @staticmethod
    def load(
        self,
        filepath,
        sep=',',
        filetype=None,
        names=None,
        conf_mat_key='confusion_matrix',
        *args,
        **kwargs,
    ):
        """Convenience function that loads the confusion matrix from the given
        filepath in given common formats. Loading from CSV relies on
        pandas.read_csv(*args, **kwargs).
        """
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


class ConfusionMatrices(ConfusionMatrix):
    """Multiple confusion matrices with an arbitrary tensor serving as the
    indices, where each index is a confusion matrix. The indexing tensor may
    consist of nominal values, thus using the NominalDataEncoder.

    This extends ConfusionMatrix and is a slice of the sparse ConfusionTensor
    if each of the different confusion matrices were different discrete random
    variables being compared.
    """
    def __init__(self):
        raise NotImplementedError()


class ConfusionTensor(ConfusionMatrix):
    """A Confusion Tensor of M dimensions all of length N, resulting in the
    shape `[N_1, N_2, ..., N_M]`.
    """
    def __init__(self):
        raise NotImplementedError()

    # TODO All the above measures need generalized such that they are applied
    # to a matrix slice. The idea is to generalize them for this Tensor or a
    # slice of this Tensor, thus needing to work on sparse np.ndarrays and
    # easily apply to both cases, w/ this possibly being the Parent of
    # ConfusionMatrix and ConfusionMatrices.
