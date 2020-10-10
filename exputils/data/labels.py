"""Useful label management classes and functions."""
import logging

from bidict import OrderedBidict
import numpy as np
from sklearn.utils import validation
from sklearn.preprocessing import label_binarize
# TODO decide how to handle the following improper usage of sklearn
from sklearn.preprocessing._label import _encode


def load_label_set(filepath, delimiter=None, shift=0):
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
    shift : int, optional
        Optional incrementation to the encoding values
    """
    if delimiter is None:
        with open(filepath, 'r') as openf:
            nd_enc = NominalDataEncoder(openf.read().splitlines(), shift)
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

# TODO NominalDataMap(): to allow for mapping any nominal data to other nominal
# labels, while the NominalDataEncoder is for encoding nominal data to integers
#   The point for this is to extend OrderedBidict so it already has the ability
#   to to transform numpy arrays when given.

class NominalDataEncoder(object):
    """A single class for handling the encoding of nominal data to integer
    values or one hot vectors.

    nominal_value -> int -> one_hot_vector or binary vector

    Consider extending bidict.

    Attributes
    ----------
    encoder : OrderedBidict
        The bidirectional mapping of nominal value to integer encoding. There
        can be no multiple keyes that map to the same values.

    Notes
    -----
        This is to provide ease-of-use for handling nominal data encodings
        where sklearn's label encoders were not as ergonomic. Here, "ergonomic"
        is about grouping the necessary parts together for handling labels so
        it is all in one place. This class could be implemented to wrap the
        sklearn's encoders, but Bidict was used instead along with sklearn
        functions.
    """
    # TODO perhaps inherit from sklear...LabelEncoder, given using its code.
    #   The idea was to extend OrderedBidict to do efficient numpy
    #   transformations similar to how sklearn...LabelEncoder does, and to also
    #   organize the locality of labels, their encodings, and functions for
    #   transforming data to and from the label encodings.
    #
    #   Furthermore, this is to aide in working with labels in general, esp. in
    #   the case of complex label relationships and updating and changing
    #   labels at certain levels of class hierarchy. So TODO: add ease of
    #   combining NominalDataEncoders together, and this is where shift then
    #   would come into play.
    def __init__(
        self,
        ordered_keys,
        shift=0,
        pos_label=1,
        neg_label=0,
        sparse_output=False,
        ignore_dups=False,
    ):
        """
        Parameters
        ----------
        shift : int, optional
            Shifts the encoding by the given value. Can be seen as the starting
            value of the ordered encodings.
        ignore_dups : bool, optional
            Ignores any duplicates in the given ordered keys. Not implemented!
        """
        if not ignore_dups and len(set(ordered_keys)) != len(ordered_keys):
            raise ValueError('There are duplicates in the given sequence')

        if ignore_dups:
            raise NotImplementedError('Ignore_dups is not yet implemented.')

        self.encoder = OrderedBidict(
            {key: enc + shift for enc, key in enumerate(ordered_keys, shift)}
        )

        self.pos_label = pos_label
        self.neg_label = neg_label
        self.sparse_output = sparse_output

    # TODO numpy vectorizaiton of encode/decode
    #   nominal value <=> int
    #   int <=> one hot
    #   nominal value <=> one hot/binary

    def keys(self, *args, **kwargs):
        return self.encoder.keys(*args, **kwargs)

    def values(self, *args, **kwargs):
        return self.encoder.values(*args, **kwargs)

    def items(self, *args, **kwargs):
        return self.encoder.items(*args, **kwargs)

    def encode(self, keys, one_hot=False):
        """Encodes the given values into their respective encodings.

        Parameters
        ----------
        keys : scalar or np.ndarray
        one_hot : bool
            If True, then expects to encode the keys into their respective one
            hot vectors. Otherwise, expects to map elements to their respective
            encoding values.

        Returns
        -------
        scalar or np.ndarray
            Same shape as input keys, but with elements changed to the proper
            encoding.
        """
        # TODO real tempted to make it so this done through
        # OrderedBidict.__getitem__(), only issue is setting values w/ = when
        # OrderedBidict[np.ndarray] = some value, which is probably
        # functionality we do not want. May be confusion too.

        if one_hot:

            return label_binarize(
                keys,
                classes=np.asarray(self.keys()),
                pos_label=self.pos_label,
                neg_label=self.neg_label,
                sparse_output=self.sparse_output,
            )

        keys = validation.column_or_1d(keys, warn=True)

        if validation._num_samples(keys) == 0:
            return np.array([])

        _, keys = _encode(keys, uniques=np.asarray(self.keys()), encode=True)

        return keys

    def decode(self, encodings, one_hot=False):
        """Decodes the given encodings into their respective keys.

        Parameters
        ----------
        encodings : scalar or np.ndarray
        one_hot : bool
            If True, then expects to decode one hot vectors into their
            respective keys. Otherwise, expects to map elements to their
            respective keys.

        Returns
        -------
        scalar or np.ndarray
            Same shape as input encodings, but with elements changed to the
            proper encoding.
        """
        # TODO real tempted to make it so this done through
        # OrderedBidict.inverse.__getitem__()

        encodings = validation.column_or_1d(encodings, warn=True)
        # inverse transform of empty array is empty array
        if validation._num_samples(encodings) == 0:
            return np.array([])

        diff = np.setdiff1d(encodings, np.arange(len(self.keys())))
        if len(diff):
            raise ValueError(
                "encodings contains previously unseen labels: %s" % str(diff)
            )
        encodings = np.asarray(encodings)
        return np.asarray(self.keys())[encodings]

    def shift_encoding(self, shift):
        """Increments or decrements all encodings by the given integer.

        Parameters
        ----------
        shift : int
            shifts all encodings by this constant integer.
        """
        if not isinstance(shift, int):
            raise TypeError(' '.join([
                'Expected `adjustment` to be type `int`, not',
                'f`{type(adjustment)}`',
            ]))

        # NOTE uncertain when shift comes into play outside of maintence or
        # when a enc value that is off from that of array indices applies.

        if shift == 0:
            logging.debug('Shift value given was zero. No shifting done.')
            return

        for key in self.encoder:
            self.encoder[key] += shift

    def append(self, keys):
        """Appends the keys to the end of the encoder giving them their
        respective encodings.
        """
        last_enc = next(reversed(self.encoder.inverse))

        if (
            isinstance(keys, list)
            or isinstance(keys, tuple)
            or isinstance(keys, np.ndarray)
        ):
            # Add the multiple keys to the encoder in order.
            for key in keys:
                if key not in self.encoder:
                    last_enc += 1
                    self.encoder[key] = last_enc
                else:
                    # NOTE could add optional ignore_dups to avoid raising
                    raise KeyError(
                        f'Given key `{key}` is already in the NominalDecoder!',
                    )
        else:
            # Add individual key
            if keys not in self.encoder:
                self.encoder[keys] = last_enc + 1
            else:
                # NOTE could add optional ignore_dups to avoid raising
                raise KeyError(
                    f'Given key `{key}` is already in the NominalDecoder!',
                )

    def reorder(self, keys):
        """Reorder the keys"""
        raise NotImplementedError()

        # TODO reorder by new sequence of keys (equivalent to making a new
        # NDEnc but preserving the shift, if there is any, which now may be a
        # depracted thing anyways, so reorder would be superfulous in this case

        # partial reorder, as in swapping class locations, may still be useful.

    def pop(self, key, encoding=False):
        """Pops the single key and updates the encoding as necessary."""
        # NOTE pop key, but then requires updating the rest of the following
        # keys, while if this was done by a list, it would be handled by
        # shifting the array and index mapping done automatically... but then
        # again, iirc, the index mapping runs into a similar issue wrt to
        # getting the index of the keys.

        # Handle the shift in encoding if there is any.
        shift = next(iter(self.encoder.inverse))

        # Obtain the last encoding
        last_enc = next(reversed(self.encoder.inverse))

        # Remove the given key, whether it is a key or encoding
        if encoding:
            self.encoder.inverse.pop(key)
            enc = key
        else:
            enc = self.encoder.pop(key)

        if enc != last_enc:
            # Decrement all following keys by one
            for k in list(self.encoder)[enc - shift:]:
                self.encoder[k] -= 1

        # TODO efficiently handle the popping of a sequence of keys and the
        # updating of the encoding.

    # TODO consider an insert_after(), inplace of a append() then reorder()


# TODO SparseNominalDataEncoder()
#   Same thing but encoding integers can jump between values, meaning missing
#   values are expected.... dunno when this is necessary
# TODO consider the following, but it is only necessary if there are holes
# in the encoder, which there should never be. Probs best to leave to user
# or child class.
#max_enc : int
#    The maximum integer encoding in the current encoder
#min_enc : int
#    The minimum integer encoding in the current encoder



# TODO class OrdinalDataEncoder: an ordinal data version
