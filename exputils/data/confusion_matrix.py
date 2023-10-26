"""Given this has been a reoccuring trend where the confusion matrix needs
calculated and certain metrics need calculated FROM the confusion matrix, add
__tested__ metrics derived from the confusion matrix for efficient
computation.
"""
from copy import copy, deepcopy
import os

import h5py
import numpy as np
import pandas as pd
from plotly import graph_objects as go
from scipy.stats import gmean, entropy

# TODO refactor: Use of Sklearn is not necessary. I had to implement top-k cms,
# this is just top-1
from sklearn.metrics import confusion_matrix

from exputils.io import create_filepath
from exputils.data.labels import NominalDataEncoder as NDE

import logging
logger = logging.getLogger(__name__)

# TODO ConfusionTensor: generalize confusion matrix to multiple discrete RVs
# with a ConfusionTensor class, which would be the parent to ConfusionMatrix

# TODO consider ways to interlink this with LabelEncoder better. If necessary


class ConfusionMatrix(object):
    """Confusion matrix of occurrences for nominal data.
    Rows are the known labels and columns are the predictions.

    Attributes
    ----------
    mat : np.ndarray
        The confusion matrix. TODO make mat private marked: _mat
    label_enc : NominalDataEncoder = None
    """
    def __init__(self, targets, preds=None, labels=None, *args, **kwargs):
        """
        Parameters
        ----------
        targets : np.ndarray
            The confusion matrix to wrap, or the target labels.
        preds : np.ndarray = None
        labels : np.ndarray | NominalDataEncoder = None
            The labels for the row and columns of the confusion matrix.
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

                if isinstance(labels, NDE):
                    self.label_enc = labels
                elif labels is not None:
                    self.label_enc = NDE(labels)
                elif isinstance(targets, pd.DataFrame):
                    self.label_enc = NDE(targets.columns)
                else:
                    self.label_enc = NDE(np.arange(len(self.mat)))
            else:
                raise TypeError(' '.join([
                    'targets type is expected to be of type `np.ndarray`, but',
                    f'recieved type of {type(targets)}.',
                ]))
        elif preds is not None:
            # Calculate the confusion matrix from targets and preds with sklearn
            if isinstance(labels, NDE):
                self.label_enc = labels
            elif isinstance(labels, (list, np.ndarray, dict)):
                self.label_enc = NDE(labels)
            else:
                self.label_enc = NDE(list(set(targets) | set(preds)))

            # TODO refactor: Use of Sklearn is not necessary. I had to
            # implement top-k cms, this is just top-1
            self.mat = confusion_matrix(
                targets,
                preds,
                labels=self.labels,
                *args,
                **kwargs,
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
        """Add two ConfusionMatrices together if of the same shape w/ same
        label set."""
        return self.join(other, 'left', False)

    @property
    def labels(self):
        if self.label_enc is not None:
            return np.array(self.label_enc)

    def join(self, other, method='left', inplace=False):
        """Joins this Confusion Matrix with another using a set method over
        their labels.

        Current version always combines with left union. First, we get the
        intersecting, left (self) disjoint, and right (other) disjoint label
        sets. Then, we work off of the self mat, adding the intersecting values
        from other into correct  spot for self mat. Adding to mat, includes
        the self disjoint label confusion already. To add the other confusion,
        the original mat is expanded with zeros to append the new labels to the
        end. This exteneded area includes three different sub matricies to be
        filled with from the other's mat. The sub mat where only other disjoint
        label confusion exists, the sub mat for disjoint other confusion with
        shared labels, and the inverse, shared labels confusion with the
        disjoint other labels. This covers all the sub mat blocks that from
        the new resulting confusion matrix.

        Args
        ----
        other : ConfusionMatrix
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
        if not isinstance(other, ConfusionMatrix):
            raise TypeError(
                'Operator add only supported between two ConfusionMatrices.'
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

        new_cm = self if inplace else deepcopy(self)
        n_other = len(other_disjoint)

        # Sum the mats together aligned by their shared labels.
        # First, add across the intersecting rows
        self_intersect_args = self_intersect.reshape(-1, 1).repeat(
            self_intersect.shape[0],
            axis=1,
        )
        new_cm.mat[self_intersect_args, self_intersect_args.T] += \
            other.mat[other_intersect][:, other_intersect]

        # Next, Expand mat to include other's disjoint labels.
        # Add Zeros placeholders for other's disjoint labels
        new_cm.mat = np.vstack((
            np.hstack((new_cm.mat, np.zeros([new_cm.mat.shape[0], n_other]))),
            np.zeros([n_other, new_cm.mat.shape[1] + n_other]),
        ))

        # Update the submat for other's disjoint labels with their values
        # Other's disjoint only sub mat: disjoint x disjoint
        if n_other > 0:
            new_cm.mat[-n_other:, -n_other:] = \
                other.mat[other_disjoint][:, other_disjoint]

            # Other's disjoint x intersect; and intersect x disjoint
            new_cm.mat[-n_other:, self_intersect] = \
                other.mat[other_disjoint][:, other_intersect]
            new_cm.mat[self_intersect, -n_other:] = \
                other.mat[other_intersect][:, other_disjoint]

        # Add other's disjoint labels to self label encoder
        new_cm.label_enc.append(other.label_enc.decode(other_disjoint))

        if not inplace:
            return new_cm

    def reduce(
        self,
        labels,
        reduced_label,
        inverse=False,
        inplace=False,
        reduced_idx=-1,
    ):
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
        reduced_idx : int = -1
            Integer index of where to put the reduced label. Currently only 0
            and -1 are supported for beginning or end respectively.
            TODO support if reduced_label in labels, use that index.

        Returns
        -------
        ConfusionMatrix
            The resulting reduced confusion matrix.
        """
        if reduced_idx not in {0, -1}:
            raise NotImplementedError(
                'Support for reduced_idx being an integer within [0, '
                'len(self.labels)] is not yet supported. Please use 0 for '
                'beginning and -1 for the end.'
            )
        if inverse:
            if reduced_label in self.labels and reduced_label in labels:
                labels = np.delete(
                    labels,
                    np.where(labels == reduced_label)[0][0],
                )
        else:
            if reduced_label in self.labels and reduced_label not in labels:
                labels = np.append(labels, reduced_label)

        # TODO perhaps the use of the nominal label encoder would be good here.
        mask = np.zeros(len(self.labels)).astype(bool)
        for label in labels:
            mask |= self.labels == label

        if inverse:
            # Numpy mask in where= ignores False from the calculation.
            not_mask = mask
            mask = np.logical_not(mask)
        else:
            not_mask = np.logical_not(mask)

        # Construct reduced s.t. reduced label is last index
        # TODO check if np.block is 1) more readable, 2) more efficient.
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

        # TODO General change location of reduced label, not just to beginning

        new_cm = self if inplace else deepcopy(self)
        new_cm.mat = reduced_cm

        # Update new label encoder.
        for label in np.array(new_cm.label_enc)[mask]:
            if label != reduced_label:
                new_cm.label_enc.pop(label)

        #labels=np.append(self.labels[not_mask], reduced_label),
        if reduced_idx == -1:
            if reduced_label not in new_cm.label_enc:
                new_cm.label_enc.append(reduced_label)
        elif reduced_idx == 0:
            # TODO if reduced_label not in new_cm.label_enc:
            # Move from last index to first.
            new_cm.mat = np.block([
                [new_cm.mat[[-1], [-1]], new_cm.mat[-1, :-1]],
                [new_cm.mat[:-1, [-1]], new_cm.mat[:-1, :-1]],
            ])

        # TODO implement: Ensure reduced label is at location in encoder
        assert reduced_label in new_cm.label_enc
        reduced_label_enc = new_cm.label_enc.encode([reduced_label])[0]
        if reduced_idx == -1:
            assert len(new_cm.label_enc) - 1 == reduced_label_enc
        else:
            assert reduced_idx == reduced_label_enc

        if not inplace:
            return new_cm

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

    # NOTE True Negatives for one class has overlap between the other classes'
    # counts as per class it becomes the sum of everything outside that
    # classes' row and column index, naturally resulting in overlap in counts
    # to other classes true negatives.

    def false_rates(self, label):
        """Calcuate the False Positive/Negative Rates for a single label."""
        tp = self.true_positive(label)
        fp = self.false_positive(label)
        fn = self.false_negative(label)
        tn = self.mat.sum() - tp - fp - fn

        logger.debug('tp = %f', tp)
        logger.debug('tn = %f', tn)
        logger.debug('fp = %f', fp)
        logger.debug('fn = %f', fn)

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
        denominator = (
            np.sqrt(total_sqrd - np.dot(predicted, predicted))
            * np.sqrt(total_sqrd - np.dot(actual, actual))
        )
        if denominator == 0:
            return 0
        return (
            (correct_pred * self.mat.sum() - np.dot(actual, predicted)) /
            denominator
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

        raise ValueError(f'Unexpected value for `normalized`: {normalized}')

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
        conf_mat_key='confusion_matrix',
        overwrite=False,
        *args,
        **kwargs,
    ):
        """Saves the current confusion matrix to the given filepath."""
        ext = os.path.splitext(filepath)[-1]

        if ext not in {'.csv', '.tsv', '.hdf5', '.h5'}:
            raise TypeError(' '.join([
                'Expected file extention: ".csv", ".tsv", ".hdf5", or',
                f'".h5"; not  `{ext}`',
            ]))

        filepath = create_filepath(filepath, overwrite=overwrite)

        if ext == '.csv':
            pd.DataFrame(self.mat, columns=self.labels).to_csv(
                filepath,
                index=False,
                *args,
                **kwargs,
            )
        elif ext == '.tsv':
            pd.DataFrame(self.mat, columns=self.labels).to_csv(
                filepath,
                index=False,
                sep='\t',
                *args,
                **kwargs,
            )
        else: # HDF5 elif ext in {'.hdf5', '.h5'}:
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
                        logger.warning(
                            '`names` is provided while "labels" exists in the '
                            'hdf5 file! `names` is prioritized of the labels '
                            'in hdf5 file.'
                        )
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
