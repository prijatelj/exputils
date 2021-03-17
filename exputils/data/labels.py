"""Useful label management classes and functions."""
import logging

from bidict import OrderedBidict, MutableBidict
import numpy as np
from sklearn.utils import validation
from sklearn.preprocessing import label_binarize
from sortedcollections import SortedDict, ValueSortedDict


def load_label_set(filepath, sep=None, *args, **kwargs):
    """Loads the given file and reads the labels in. Expects label per line.

    Parameters
    ----------
    filepath : str
        The filepath to the file containing the labels
    sep : str, optional
        The sep character if the file contains provided encodings per
        label. Always assumes one label per line. Will assume first column is
        the original label to be encoded to the provided encoding when
        sep is not None.
    """
    if sep is None:
        with open(filepath, 'r') as openf:
            nd_enc = NominalDataEncoder(
                openf.read().splitlines(),
                *args,
                **kwargs,
            )
        return nd_enc

    # TODO load as csv or tsv, or YAML
    raise NotImplementedError(
        f'sep is not None, but loading CSV not implemented. {sep}',
    )


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


class KeySortedBidict(MutableBidict):
     __slots__ = ()
     _fwdm_cls = SortedDict
     _invm_cls = ValueSortedDict
     _repr_delegate = list

     # TODO the KeySortedBiDict's SortedDict and ValueSortedDict are not
     # Reversible; unable to obtain reverse iterator via reversed(). This needs
     # fixed. For now use OrderedBidict


class NominalDataEncoder(object):
    """A single class for handling the encoding of nominal data to integer
    values or one hot / binarized vectors.

    nominal_value -> int -> one_hot_vector or binary vector

    Consider extending bidict.

    Attributes
    ----------
    encoder : OrderedBidict
        The bidirectional mapping of nominal value to integer encoding. There
        can be no multiple keyes that map to the same values.
    argsorted_keys : np.ndarray(int)
        When the keys in the encoder are not sorted, but instead saved in the
        order they are given, then sorted_keys_args is an array of the indices
        of the encoder keys in sorted order. This is necessary for encoding
        using numpy only when the keys are not sorted when saved into the
        encoder. If the keys are sorted when the encoder is created, then this
        is None.

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
        sort_keys=False,
        #unknown=None,
        #unknown_idx=None,
        #unknown_key='unknown_default',
    ):
        """
        Parameters
        ----------
        ordered_keys : Iterable
            The keys to be added to the
        shift : int, optional
            Shifts the encoding by the given value. Can be seen as the starting
            value of the ordered encodings.
        pos_label : int
            The positive label to use when binarizing or one hot encoding.
        neg_label : in
            The negative label to use when binarizing or one hot encoding.
        sparse_output : bool
            ??? same as scikit LabelBinarizer learn atm.
        ignore_dups : bool, optional
            Ignores any duplicates in the given ordered keys. Not implemented!
        sort_keys : bool, optional
        """
        if not ignore_dups and len(set(ordered_keys)) != len(ordered_keys):
            raise ValueError('There are duplicates in the given sequence')

        if ignore_dups:
            raise NotImplementedError('Ignore_dups is not yet implemented.')

        if sort_keys:
            # Sort the keys so they are in the encoder sorted, rather than
            # order given.
            ordered_keys = np.unique(ordered_keys)  # np.unique sorts the keys
            self.argsorted_keys = None

            # KeySortedBidict keeps the keys sorted
            self.encoder = KeySortedBidict({
                key: enc + shift for enc, key in enumerate(ordered_keys, shift)
            })
        else:
            # Use in order given, but maintain sorted_args for encoding
            unique, self.argsorted_keys = np.unique(
                ordered_keys,
                return_index=True,
            )
            # TODO probably can make ignore dups and error raise more efficient
            # if already using unique like this. NOTE that unique here does not
            # work to get argsorted_keys unless ordered_keys is already unique
            # keys only, which is it given the check and NOT ignore_dups

            self.encoder = OrderedBidict({
                key: enc + shift for enc, key in enumerate(ordered_keys, shift)
            })

        self.pos_label = pos_label
        self.neg_label = neg_label
        self.sparse_output = sparse_output

        # TODO need to flesh out the handling of unknowns in the enecoder
        """
        if unknown == 'update':
            self.unknown_idx = unknown_idx
            # TODO further functionality required in [en/de]code to handle this
            # TODO is there a default unknown or no?
            self.unknown_key = unknown_key
        elif unknown == 'single':
            self.unknown_idx = unknown_idx
            self.unknown_key = unknown_key
            # TODO further functionality required in [en/de]code to handle this
            # TODO optionally separate unknowns from the encoding, esp. in
            # one_hots
        elif unknown is not None:
            raise ValueError(' '.join([
                'Expected `unknown` to be `None`, "update", or "single", but',
                f'"recieved": {unknown}',
            ]))
        else:
            self.unknown_idx = None
        self.unknown = unknown
        #"""

    @property
    def keys_sorted(self):
        return isinstance(self.encoder, KeySortedBidict)

    #@property
    #def unknown_key(self):
    #    if self.unknown is None:
    #        raise ValueError('`unknown` is None. No unknown key or encoding!')
    #    elif self.unknown_idx is None:
    #        raise ValueError(
    #            '`unknown_idx` is None. No default unknown key or encoding!'
    #        )
        return self.encoder.inverse[self.unknown_idx]

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
        if one_hot:
            return label_binarize(
                keys,
                classes=np.array(self.encoder),
                pos_label=self.pos_label,
                neg_label=self.neg_label,
                sparse_output=self.sparse_output,
            )

        keys = validation.column_or_1d(keys, warn=True)

        if validation._num_samples(keys) == 0:
            return np.array([])

        # Check for unrecognized keys # TODO may be able to be more efficient?
        diff = set(np.unique(keys)) - set(self.encoder)
        if diff:
            #   unknowns=None
            raise ValueError(f'`keys` contains previously unseen keys: {diff}')
            # TODO allow for assigning a default encoding value if unknown
            # label: i.e. not in the current encoder
            #   unknowns=default; unknown_idx = 0

            # TODO XOR allow for updating of the labels in order of occurrence.
            # XOR default is as is, fail if unseen label in encoding.
            #   unknowns=update

        if keys.dtype == object:
            # Python encode

            return np.array([self.encoder[key] for key in keys])

        # Numpy encode
        if self.keys_sorted:
            # Encoder keys are already sorted within the encoder.
            return np.searchsorted(self.encoder, keys)

        return self.argsorted_keys[np.searchsorted(
            self.encoder,
            keys,
            sorter=self.argsorted_keys,
        )]

        # TODO to get this to work w/ np.searchsorted as sklearn does it, a
        # sorted args of the keys must always be present. This means as the
        # keys change, this sorted args must also change. Otherwise, this needs
        # done a different way. This is the cost of having any order of keys.

        #return keys

    def decode(self, encodings, one_hot_axis=None):
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
        if isinstance(one_hot_axis, int):
            encodings = encodings.argmax(axis=one_hot_axis)
            # TODO check encodings.shape to expected shape

        encodings = validation.column_or_1d(encodings, warn=True)
        # inverse transform of empty array is empty array
        if validation._num_samples(encodings) == 0:
            return np.array([])

        diff = np.setdiff1d(encodings, np.arange(len(self.keys())))
        if len(diff):
            raise ValueError(
                "encodings contains previously unseen labels: %s" % str(diff)
            )
            # TODO hard to handle unknowns in the decoding case, but could do
            # update or default as well, I suppose.

        return np.array(self.encoder)[np.array(encodings)]

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

    def append(self, keys, ignore_dups=False):
        """Appends the keys to the end of the encoder giving them their
        respective encodings.
        """
        # TODO handle the update to argsorted_keys, more efficiently
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

                    if not self.keys_sorted:
                        # Must update the argsorted_keys for approriate
                        # encoding TODO replace this hotfix cuz this is
                        # inefficient!
                        self.argsorted_keys = np.argsort(self.encoder)

                elif ignore_dups:
                    continue
                else:
                    raise KeyError(
                        f'Given key `{key}` is already in the NominalDecoder!',
                    )
        else:
            # Add individual key
            if keys not in self.encoder:
                self.encoder[keys] = last_enc + 1

                if not self.keys_sorted:
                    # Must update the argsorted_keys for approriate encoding
                    # TODO replace this hotfix cuz this is inefficient!
                    self.argsorted_keys = np.argsort(self.encoder)

            elif ignore_dups:
                return
            else:
                raise KeyError(
                    f'Given key `{keys}` is already in the NominalDecoder!',
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

        # TODO handle the update to argsorted_keys

        # Handle the shift in encoding if there is any.
        shift = next(iter(self.encoder.inverse))

        # Obtain the last encoding
        last_enc = next(reversed(self.encoder.inverse))

        if not self.keys_sorted:
            # Must remove the key's respective arg from argsorted_keys
            arg = np.argwhere(np.array(self.encoder) == (
                self.encoder.inverse[key] if encoding else key
            ))[0][0]

            self.argsorted_keys = np.delete(self.argsorted_keys, arg)

            # adjust the rest of the args accordingly
            self.argsorted_keys[np.where(self.argsorted_keys > arg)] -= 1

        # Remove the given key, whether it is a key or encoding
        if encoding:
            enc = key
            key = self.encoder.inverse.pop(key)
        else:
            enc = self.encoder.pop(key)

        if enc != last_enc:
            # Decrement all following keys by one
            for k in list(self.encoder)[enc - shift:]:
                self.encoder[k] -= 1

        return key if encoding else enc

        # TODO efficiently handle the popping of a sequence of keys and the
        # updating of the encoding.

    # TODO consider an insert_after(), inplace of a append() then reorder()

    def save(self, filepath, sep=None):
        """Saves the labels as an ordered list where the index is implied by
        the order of the labels.
        """
        if sep is None:
            with open(filepath, 'w') as openf:
                openf.write('\n'.join([str(x) for x in self.encoder]))
        else:
            raise NotImplementedError(' '.join([
                'Saving as any file using separators other than newlines',
                'between the labels is not yet supported.',
            ]))

    @staticmethod
    def load(filepath, sep=None, *args, **kwargs):
        """Loads the ordered list from the file. Defaults to expect a text file
        where each line contains a single nominal label.
        """
        return load_label_set(filepath, sep, *args, **kwargs)


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
