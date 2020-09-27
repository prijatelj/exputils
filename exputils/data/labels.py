"""Useful label management classes and functions."""
from bidict import bidict


def load_label_set(filepath, delimiter=None, increment_enc=None):
    """Loads the given file and reads the labels in. Expects label per line.

    Parameters
    ----------
    filepath : str
        The filepath to the file containing the labels
    delmiter : str, optional
        The delimiter character if the file contains provided encodings per
        label. Always assumes one label per line. Will assume first column is
        the original label to be encoded to the provided encoding when
        delimiter is not None.
    increment_enc : int, optional
        Optional incrementation to the encoding values
    """
    if increment_inc is not None:
        raise NotImplementedError('incrementing the encoding integer values.')

    if delimiter is None:
        with open(filepath, 'r') as openf:
            nd_enc = NominalDataEncoder(openf.read().splitlines())
        return nd_enc

    # TODO load as csv or tsv


class BidirDict(dict):
    """A bidirectional dictionary that streamlines the maintenance of two
    dictionaries used together and allows for keys to map to multiple of the
    same values.

    Notes
    -----
        This is modified from the original code provided at:
            https://stackoverflow.com/a/21894086/6557057
    """
    def __init__(self, *args, **kwargs):
        super(BidirDict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value,[]).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(BidirDict, self).__setitem__(key, value)
        self.inverse.setdefault(value,[]).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key],[]).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(BidirDict, self).__delitem__(key)


# TODO perhaps make a NominalDataBidict and have the encoders be numeric
# encodings only? The Bidict would allow for ease for changing labels to others
# but perhaps only bidict and BidirDict are necessary for that, rather than its
# own class. That depends on what else is desired by the NominalDataBidict


class NominalDataEncoder(object):
    """A single class for handling the encoding of nominal data to integer
    values or one hot vectors.

    nominal_value -> int -> one_hot_vector or binary vector

    Consider extending bidict.

    Attributes
    ----------
    encoder : bidict
        The bidirectional mapping of nominal value to integer. There can be no
        multiple keyes that map to the same values.


    Notes
    -----
        This is to provide ease-of-use for handling nominal data encodings
        where scikit learn's label encoders were not as ergonomic.
    """
    def __init__(self, sequence, ):
        # TODO init w/ default mapping, provided mapping, provided start index

    # TODO ease update of mapping,

    # TODO numpy vectorizaiton of encode/decode
    #   nominal value <=> int
    #   int <=> one hot
    #   nominal value <=> one hot/binary

    def keys(self, *args, **kwargs):
        return self.encoder.keys()

    def values(self, *args, **kwargs):
        return self.encoder.values()

    def items(self, *args, **kwargs):
        return self.encoder.items()

    def encode(self, values):
        """Encodes the given values into their respective encodings.

        Parameters
        ----------
        values : scalar or np.ndarray
        """

        return

    def decode(self, encodings):
        """Decodes the given encodings into their respective values.

        Parameters
        ----------
        encodings : scalar or np.ndarray
        """

        return

    def adjust_encoding(self, adjustment):
        """Increments or decrements all encodings by the given integer."""
        if not isinstance(adjustment, int):
            raise TypeError(' '.join([
                'Expected `adjustment` to be type `int`, not,
                f`{type(adjustment)}`',
            )
        # TODO implement adjustment of all encodings
        raise NotImplementedError('No adjusting the encoding integer values.')


    def update(self, *args, **kwargs):
        """Updates the encoder with the given values."""
        # TODO updates the bidict encoder with the given values
        self.encoder.update(*args, **kwargs)


# TODO class OrdinalDataEncoder: an ordinal version
