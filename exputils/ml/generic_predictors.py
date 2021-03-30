"""Abstract or Generic classes for predictors of different types. Intended to organize
creation of predictors.
"""
from abc import ABC, abstractmethod
import pickle
import gzip

import numpy as np

from exputils.data.labels import NominalDataEncoder
from exputils.io import create_filepath


class Stateful(object):
    """Generic class for a stateful object who needs save and load methods."""
    def save(self, filepath, overwrite=False):
        """Given filepath, save the current state of the object."""
        # TODO attempt generic compression and saving of python object using
        # more efficent compression for numerical data w/o the need for
        # serialization, e.g. hdf5 or numpy npz. Otherwise, fallback to
        # compressed pickle (unnecessary in general case, consider separate
        # method to call for when it is necessary, e.g. complicated models)

        # TODO if diff saving used, chose via file ext, and set ext if none
        # after attempting above and determining which is used.

        # Generic save using pickle with gzip compression
        with gzip.open(create_filepath(filepath, overwrite), 'wb') as openf:
            openf.write(pickle.dumps(self))

        # TODO add option to save params to json/yaml . . . ? (opt. compress)
        #   At least add to dict (vars) and __str__ version

        # TODO make the different generic saves/loads be their own functions
        # that are called here and then may be called elsewhere too.

    @staticmethod
    def load(filepath):
        """Given filepath, load the saved state of the object."""
        # Generic load using pickle
        with gzip.open(filepath, 'rb') as openf:
            return pickle.load(self, openf.read())


class Predictor(Stateful):
    """Abstract class for predictors."""
    @abstractmethod
    def predict(self, features):
        """Given the current state of the predictor, predict the labels"""

        # TODO predict in batches
        raise NotImplementedError()


class SupervisedLearner(Predictor):
    """Abstract class for supervised learning predictors."""

    # TODO __init__ either sets random state or setups up random state to be
    # created when first fit occurs (implies unable to predict in this case).
    # assumes stochastic learning.

    @abstractmethod
    def fit(self, features, labels):
        """Given the current state of the predictor, continue training given
        the provided data. This uses the existing state of the predictor.
        """

        # TODO fit in batches
        # TODO fit incrementally
        # TODO parameterize init, fit, and predict st they may be provided args
        # & kwargs as configs
        raise NotImplementedError()


class SupervisedClassifier(SupervisedLearner):
    """Abstract class for supervised learning classifiers.

    Attributes
    ----------
    label_enc : NominalDataEncoder
    """

    # TODO __init__ either sets random state or setups up random state to be
    # created when first fit occurs (implies unable to predict in this case).

    def __init__(self, *args, **kwargs):
        self.label_enc = NominalDataEncoder(*args, **kwargs)

    @property
    def labels(self):
        return np.array(self.label_enc.encoder)

# TODO numpy, python, tensorflow, torch, etc. RandomState class that serves as
# a single randomstate object and updates whenever it is used by any of these
# libs, keeping it consistent and easy to manage in a cross lib project.
# Someone may have done this already.
