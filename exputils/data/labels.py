"""Useful label management classes and functions."""
from copy import deepcopy
from itertools import islice
import numbers
import os
import re

from bidict import OrderedBidict, MutableBidict
import h5py
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import _num_samples as n_samples

from exputils.io import create_filepath

import logging
logger = logging.getLogger(__name__)


# TODO add Torch version of this: support for doing this in Torch code only
# TODO add Tensorflow version of this: support for doing this in TF code only
# TODO add JAX version of this: support for doing this in JAX code only
# TODO If it makes sense to, break dep on sklearn


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


"""
class KeySortedBidict(MutableBidict):
    __slots__ = ()
    _fwdm_cls = SortedDict
    _invm_cls = ValueSortedDict
    _repr_delegate = list

    # TODO the KeySortedBiDict's SortedDict and ValueSortedDict are not
    # Reversible; unable to obtain reverse iterator via reversed(). This needs
    # fixed. For now use OrderedBidict
#"""


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
    _argsorted_keys : np.ndarray(int) = None
        When the keys in the encoder are not sorted, but instead saved in the
        order they are given, then _argsorted_keys is an array of the indices
        of the encoder keys in sorted order. This is necessary for encoding
        using numpy only when the keys are not sorted when saved into the
        encoder. If the keys are sorted when the encoder is created, then this
        is None.

    self._argsorted_unknown_idx : int = None
    pos_label : int = 1
        The positive label to use when binarizing or one hot encoding.
    neg_label : int = 0
        The negative label to use when binarizing or one hot encoding.
    sparse_output : bool = False
        ??? same as scikit LabelBinarizer learn atm.
    _unknown_key : str = None
        The key used for an unknown label, especially in encoding of unknown
        or other labels not contained within the encoder. When this is None,
        the default, no handling of unknowns is supported by this encoder.

        This class is different from the others in that nominal values
        encountered in encoding that are not in the set of known labels will be
        treated as this unknown label.

    Notes
    -----
        This is to provide ease-of-use for handling nominal data encodings
        where sklearn's label encoders were not as ergonomic. Here, "ergonomic"
        is about grouping the necessary parts together for handling labels so
        it is all in one place. This class could be implemented to wrap the
        sklearn's encoders, but Bidict was used instead along with sklearn
        functions.
    """
    def __init__(
        self,
        ordered_keys=None,
        shift=0,
        pos_label=1,
        neg_label=0,
        sparse_output=False,
        ignore_dups=False,
        sort_keys=False,
        unknown_key=None,
        unknown_idx=None,
    ):
        """
        Parameters
        ----------
        ordered_keys : str
            any Iterable
            The keys to be added to the encoder.
        shift : int = 0
            Shifts the encoding by the given value. Can be seen as the starting
            value of the ordered encodings.
        pos_label : see self
        neg_label : see self
            The negative label to use when binarizing or one hot encoding.
        sparse_output : see self
        ignore_dups : bool = False
            Ignores any duplicates in the given ordered keys. Not implemented!
        sort_keys : bool = False
        unknown_key : see self
        unknown_idx : int = None
            The index encoding for the unknown catch-all class, which is only
            used when `unknown_key` is not None. When not given, this defaults
            to shift, which defaults to 0.

            If storing the unknown key within the encoder, we recommend having
            unknown_idx = 0 when used to be represented as some other or
            unknown class that is not captured by the rest of the classes,
            especially when new known classes may be added to the encoder, as
            this keeps unknown consistently at the same indedx, regardless of
            prior or future known classes.
        """
        if ordered_keys is None:
            ordered_keys = []
        if not ignore_dups and len(set(ordered_keys)) != len(ordered_keys):
            raise ValueError('There are duplicates in the given sequence')

        if ignore_dups:
            raise NotImplementedError('Ignore_dups is not yet implemented.')

        if neg_label >= pos_label:
            raise ValueError(
                'neg_label ({neg_label}) >= pos_label ({pos_label}), must be <'
            )
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.sparse_output = sparse_output

        if sort_keys:
            # np.unique sorts the keys lexically
            ordered_keys = np.unique(ordered_keys)
        self._argsorted_keys = None
        self._argsorted_unknown_idx = None

        """TODO
        # Sort the keys so they are in the encoder sorted, rather than
        # order given.

        # KeySortedBidict keeps the keys sorted when updated.
        self.encoder = KeySortedBidict({
            key: enc + shift for enc, key in enumerate(ordered_keys, shift)
        })
        #"""
        #else:
        # Use in order given, but maintain sorted_args for encoding
        # TODO probably can make ignore dups and error raise more efficient
        # if already using unique like this. NOTE that unique here does not
        # work to get _argsorted_keys unless ordered_keys is already unique
        # keys only, which is it given the check and NOT ignore_dups

        self.encoder = OrderedBidict({
            key: enc for enc, key in enumerate(ordered_keys, shift)
        })

        # Handle all unknown key internal state and checks
        # TODO create method: self.set_unknown(key, idx)
        self._unknown_key = unknown_key
        if unknown_key is not None:
            """
            raise NotImplementedError(' '.join([
                'init setup, but encode needs checked for non object dtypes!',
                'The python encode object dtype works cuz of dict get default',
                'If ints or floats, the unknowns will not default to unknown',
            ]))
            #"""
            if unknown_key in self.encoder:
                if unknown_idx is not None:
                    # Check if the unknown index is correct
                    if unknown_idx != self.encoder[self.unknown_key]:
                        raise ValueError(' '.join([
                            '`unknown_key` in ordered_keys but `unknown idx`',
                            'given a different index than expected:',
                            f'`unknown_idx` is {unknown_idx}, but expected',
                            f'{self.unknown_key} based on ordered keys.',
                        ]))
            elif unknown_idx in {None, shift}:
                self.shift_encoding(1)
                tmp = OrderedBidict({unknown_key: shift})
                tmp.update(self.encoder)
                self.encoder = tmp
            elif unknown_idx in {-1, len(self.encoder) + shift}:
                self.append(unknown_key)
                # TODO unit test to ensure append handles unknown correctly!
            elif (
                isinstance(unknown_idx, int)
                and unknown_idx >= shift
                and unknown_idx <= shift + len(self.encoder)
            ):
                first_half = OrderedBidict(
                    islice(self.encoder.items(), unknown_idx - shift)
                )
                first_half.update(OrderedBidict({unknown_key: unknown_idx}))
                first_half.update(OrderedBidict({
                    k : v + 1 for k, v in
                    islice(self.encoder.items(), unknown_idx - shift, None)
                }))
                self.encoder = first_half

            else:
                raise TypeError(' '.join([
                    '`unknown_idx` is not None or an int within label indices',
                    f'[{shift}, {shift + len(self.encoder)}]!',
                ]))

            #if self._argsorted_keys is not None:

        if not sort_keys:
            # Must update the _argsorted_keys for approriate encoding
            # TODO replace this hotfix cuz this is inefficient!
            #self._argsorted_keys = np.argsort(self.encoder)
            self._update_argsorted_keys(self.encoder)

    def __eq__(self, other):
        if not isinstance(other, NominalDataEncoder):
            logger.warning(
                'Unsupported comparison of NominalDataEncoder object to an '
                'object of type `%s`. Performing surface level duck typing.',
                type(other),
            )
        for key in self.__dict__:
            self_val = getattr(self, key)
            other_val = getattr(other, key)
            try:
                if self_val != other_val:
                    return False
            except ValueError:
                if (self_val != other_val).any():
                    return False
        return (
            self.unknown_key == other.unknown_key
            and self.unknown_idx == other.unknown_idx
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, obj):
        return obj in self.encoder

    def __getitem__(self, key):
        return self.encoder[key]

    def __iter__(self):
        return iter(self.encoder)

    def __reversed__(self):
        return reversed(self.encoder)

    def __len__(self):
        return len(self.encoder)

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

    def _update_argsorted_keys(self, keys):
        """Updates argsorgted keys and the argsorted idx of the first key."""
        unique, self._argsorted_keys = np.unique(keys, return_index=True)
        if self.unknown_key is not None:
            self._argsorted_unknown_idx = np.flatnonzero(
                [uni == self.unknown_key for uni in unique]
            )[0]

    @property
    def inv(self):
        return self.encoder.inverse

    @property
    def inverse(self):
        return self.encoder.inverse

    @property
    def are_keys_sorted(self):
        """Returns Boolean of if the keys are sorted within this encoder."""
        return self._argsorted_keys is None

    @property
    def unknown_key(self):
        return self._unknown_key

    @property
    def unknown_idx(self):
        return self.encoder.get(self._unknown_key, None)

    @property
    def shift(self):
        return next(iter(self.encoder.inverse))

    def keys(self, *args, **kwargs):
        return self.encoder.keys(*args, **kwargs)

    def values(self, *args, **kwargs):
        return self.encoder.values(*args, **kwargs)

    def items(self, *args, **kwargs):
        return self.encoder.items(*args, **kwargs)

    def get(self, key, default=None):
        return self.encoder.get(key, default)

    def encode(self, keys, one_hot=False):
        """Encodes the given values into their respective encodings.

        Parameters
        ----------
        keys : scalar or np.ndarray
        one_hot : bool = False
            If True, then expects to encode the keys into their respective one
            hot vectors. Otherwise, expects to map elements to their respective
            encoding values.

        Returns
        -------
        scalar or np.ndarray
            Same shape as input keys, but with elements changed to the proper
            encoding.

        Notes
        -----
        When unknown_idx is not None, currently uses numpy.where() after
        checking all np.searchsorted values of 0 and len(self.encoder) and
        checks all values in keys for where the np.seachsorted 0 is an unknown
        label. This is probably inefficient and could be improved.

        TODO, may be beneficial to include a live updating behavior that when
        an unknown token is encountered, rather than throwing an error if
        unknown_key does not exist, or treating that token as the unknown key,
        update the encoder to include this new token. Could be a bool param
        `update` that defaults to False here.
        """
        if len(self) == 0:
            raise ValueError('NominalDataEncoder is empty. Unable to encode.')
        if one_hot:
            try:
                return label_binarize(
                    keys,
                    classes=np.array(self.encoder),
                    pos_label=self.pos_label,
                    neg_label=self.neg_label,
                    sparse_output=self.sparse_output,
                )
            except ValueError:
                pass

        keys = np.array(keys)

        if n_samples(keys) == 0:
            return np.array([])

        # Check for unrecognized keys # TODO may be able to be more efficient?
        diff = set(np.unique(keys)) - set(self.encoder)
        if diff:
            #   unknowns=None
            if not self._unknown_key:
                raise ValueError(
                    f'`keys` contains previously unseen keys: {diff}',
                )
            #else:
            #    raise NotImplementedError(' '.join([
            #        f'`keys` contains previously unseen keys: {diff}',
            #        'and unknown_key is provided, but not implremented yet!',
            #    ]))

            # TODO XOR allow for updating of the labels in order of occurrence.
            # XOR default is as is, fail if unseen label in encoding.
            #   unknowns = update
            # if update: ...

        shift = self.shift

        if keys.dtype == object:
            # Python encode
            encoded =np.array([
                self.encoder.get(key, self.unknown_idx) for key in keys
            ]).reshape(keys.shape)

        # Numpy encode
        elif self.are_keys_sorted:
            # Encoder keys are already sorted within the encoder.
            encoded = np.searchsorted(self.encoder, keys)
            if self.unknown_idx is not None:
                # Checks where encoded == len(encoder) or (0 and is unknown)
                # Assigns unknown idx to that position, this handles the case
                # of ordered elements being less than all values or greater
                encoded = np.where(
                    (
                        (encoded != len(self.encoder)) # Not unknown greater
                        & ~ ( # Not unknown lesser (at beginning)
                            (encoded == 0)
                            & (keys == self.inv[shift])
                        )
                    ),
                    encoded,
                    self.unknown_idx,
                )
        else:
            # Encoder keys are not necessarily lexically sorted w/in encoder.
            encoded = np.searchsorted(
                self.encoder,
                keys,
                sorter=self._argsorted_keys,
            )
            if self.unknown_idx is None:
                encoded = self._argsorted_keys[encoded]
            else:
                # Checks where encoded == len(encoder) or (0 and is unknown)
                # Assigns unknown idx to that position, this handles the case
                # of ordered elements being less than all values or greater
                encoded = self._argsorted_keys[np.where(
                    (
                        (encoded != len(self.encoder)) # Not unknown greater
                        & ~ ( # Not unknown lesser (at beginning)
                            (encoded == 0)
                            & (keys == self.inv[shift])
                        )
                    ),
                    encoded,
                    self._argsorted_unknown_idx,
                )]

        if one_hot:
            # TODO beware shift for when a one hot encoding!
            # Supporting multioutput data: make new axis last by default
            if self.pos_label != 1 or self.neg_label != 0:
                n_classes = len(self.encoder)
                one_hot_classes = np.empty([n_classes] * 2)
                one_hot_classes.fill(self.neg_label)
                one_hot_classes[np.eye(n_classes, dtype=bool)] = \
                    np.array([self.pos_label] * n_classes)
            else:
                one_hot_classes = np.eye(len(self.encoder))
            return one_hot_classes[encoded]
            # TODO support changing where the added dimension goes.
        elif keys.dtype != object and self.shift != 0:
            logger.debug('shift = %d', self.shift)
            return encoded + self.shift
        return encoded

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
        if len(self) == 0:
            raise ValueError('NominalDataEncoder is empty. Unable to decode.')
        if isinstance(one_hot_axis, int):
            # TODO check encodings.shape to expected shape
            encodings = encodings.argmax(axis=one_hot_axis)
            if self.shift != 0:
                encodings += self.shift

        # inverse transform of empty array is empty array
        if n_samples(encodings) == 0:
            return np.array([])

        diff = np.setdiff1d(encodings, np.array(self.inv))
        if len(diff):
            raise ValueError(
                f'encodings contains previously unseen labels: {diff}'
            )
            # TODO hard to handle unknowns in the decoding case, but could do
            # update or default as well, I suppose.

        if self.shift != 0:
            return np.array(self.encoder)[np.array(encodings) - self.shift]
        return np.array(self.encoder)[np.array(encodings)]

    def shift_encoding(self, shift):
        """Increments or decrements all encodings by the given integer.

        Parameters
        ----------
        shift : int
            shifts all encodings by this constant integer.
        """
        if len(self) == 0:
            raise ValueError('NominalDataEncoder is empty, nothing to shift.')
        if not isinstance(shift, int):
            raise TypeError(' '.join([
                'Expected `adjustment` to be type `int`, not',
                f'`{type(shift)}`',
            ]))

        # NOTE uncertain when shift comes into play outside of maintence or
        # when a enc value that is off from that of array indices applies.

        if shift == 0:
            logger.debug('Shift value given was zero. No shifting done.')
            return

        if shift > 0:
            for key in reversed(self.encoder):
                self.encoder[key] += shift
        else:
            for key in self.encoder:
                self.encoder[key] += shift

    # TODO insert(key, idx): insert key at idx, moving everything at idx down 1

    def append(self, keys, ignore_dups=False):
        """Appends the keys to the end of the encoder giving them their
        respective encodings.
        """
        # TODO handle the update to _argsorted_keys, more efficiently
        if len(self) == 0:
            last_enc = -1
        else:
            last_enc = next(reversed(self.encoder.inverse))

        if isinstance(
            keys,
            (list, tuple, set, np.ndarray, NominalDataEncoder),
        ):
            # Add the multiple keys to the encoder in order.
            for key in keys:
                if key not in self.encoder:
                    last_enc += 1
                    self.encoder[key] = last_enc

                    if not self.are_keys_sorted:
                        # Must update the _argsorted_keys for approriate
                        # encoding TODO replace this hotfix cuz this is
                        # inefficient! # TODO unit test this!
                        #self._argsorted_keys = np.argsort(self.encoder)
                        self._update_argsorted_keys(self.encoder)
                elif not ignore_dups:
                    raise KeyError(
                        f'Given key `{key}` is already in the NominalDecoder!',
                    )
        else:
            # Add individual key
            if keys not in self.encoder:
                self.encoder[keys] = last_enc + 1

                if not self.are_keys_sorted:
                    # Must update the _argsorted_keys for approriate encoding
                    # TODO replace this hotfix cuz this is inefficient!
                    # TODO unit test this!
                    #self._argsorted_keys = np.argsort(self.encoder)
                    self._update_argsorted_keys(self.encoder)
            elif not ignore_dups:
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
        if len(self) == 0:
            raise KeyError(f'{key}')
        if key == self.unknown_key:
            self._unknown_key = None
            self._argsorted_unknown_idx = None

        # TODO handle the update to _argsorted_keys

        # Handle the shift in encoding if there is any.
        #shift = next(iter(self.encoder.inverse))

        # Obtain the last encoding
        last_enc = next(reversed(self.encoder.inverse))
        prior_shift = self.shift

        # Remove the given key, whether it is a key or encoding
        if encoding:
            enc = key
            key = self.encoder.inverse.pop(key)
        else:
            enc = self.encoder.pop(key)

        #logger.debug('key = `%s`, enc = `%s`', key, enc)

        if enc != last_enc:
            # Decrement all following keys by one
            for key in list(self.encoder)[enc - prior_shift:]:
                #logger.debug(
                #    'key = `%s`, encoded key = `%s`',
                #    key,
                #    self.encoder[key],
                #)
                self.encoder[key] -= 1

        if not self.are_keys_sorted:
            # Must remove the key's respective arg from _argsorted_keys
            """ TODO fix
            arg = np.argwhere(np.array(self.encoder) == (
                self.encoder.inverse[key] if encoding else key
            ))[0][0]

            self._argsorted_keys = np.delete(self._argsorted_keys, arg)

            # adjust the rest of the args accordingly
            self._argsorted_keys[np.where(self._argsorted_keys > arg)] -= 1
            #"""
            self._update_argsorted_keys(self.encoder)

        return key if encoding else enc

        # TODO efficiently handle the popping of a sequence of keys and the
        # updating of the encoding.

    # TODO consider an insert_after(), inplace of a append() then reorder()

    def save_h5(self, h5, overwrite=False):
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(create_filepath(h5, overwrite), 'w')

        keys_dtype = type(next(iter(self)))
        h5.attrs["keys_dtype"] = str(keys_dtype)

        if keys_dtype in {np.str_, np.string_, object, str}:
            h5.create_dataset(
                "keys",
                data=np.array(self).astype(object),
                dtype=h5py.special_dtype(vlen=str),
            )
            keys_dtype = str
        else:
            h5["keys"] = np.array(self)

        # Complete save of state with the init kwargs as attrs
        for key, val in dict(
            shift=self.shift,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
            sparse_output=self.sparse_output,
            unknown_key=keys_dtype(self.unknown_key),
            unknown_idx=self.unknown_idx,
        ).items():
            if val is None:
                continue
            h5.attrs[key] = val

        # Complete save of state with non-init attrs as values/datasets
        if self._argsorted_keys is not None:
            h5['_argsorted_keys'] = self._argsorted_keys
            if self._argsorted_unknown_idx is not None:
                h5.attrs['_argsorted_unknown_idx']=self._argsorted_unknown_idx

        if close:
            h5.close()

    def save(self, filepath, overwrite=False, sep=None):
        """Saves the labels as an ordered list where the index is implied by
        the order of the labels.
        """
        if len(self) == 0:
            raise ValueError('NominalDataEncoder is empty, nothing to save.')
        if isinstance(filepath, (h5py.Group, h5py.File)):
            self.save_h5(filepath, overwrite)
            return

        if (
            next(iter(self.inverse)) != self.shift
            or next(reversed(self.inverse)) != len(self) - 1 + self.shift
        ):
            logger.warning(
                'This does not save the encoder.values(), but the value range '
                'does not start at self.shift nor end at len(self) - 1 + '
                'shift. Order is preserved.'
            )

        ext = os.path.splitext(filepath)[-1]
        if isinstance(ext, str) and ext in {'.h5', '.hdf5'}:
            self.save_h5(filepath, overwrite)
        elif sep is None:
            with open(filepath, 'w') as openf:
                openf.write('\n'.join([str(x) for x in self.encoder]))
        else:
            raise NotImplementedError(
                'Saving as any file using separators other than newlines '
                'between the labels is not yet supported.'
            )

    @staticmethod
    def load(filepath, *args, **kwargs):
        """Loads the ordered list from the file. Defaults to expect a text file
        where each line contains a single nominal label.

        Args
        ----
        filepath : str
        sep : str = None
        """
        if (
            isinstance(filepath, str)
            and os.path.splitext(filepath)[-1].lower() not in {'.h5', '.hdf5'}
        ):
            return load_label_set(filepath, *args, **kwargs)
        return NominalDataEncoder.load_h5(filepath)

    @staticmethod
    def load_h5(h5):
        close = isinstance(h5, str)
        if close:
            h5 = h5py.File(h5, 'r')

        attrs = dict(h5.attrs.items())

        type_regex = re.compile("<class '(?P<type>.+)'>")
        keys_dtype = attrs.pop('keys_dtype', None)
        if keys_dtype:
            if keys_dtype in {str(np.str_), str(np.string_)}:
                keys_dtype = getattr(np, keys_dtype.rpartition('.')[-1][:-2])
            else:
                keys_dtype = type_regex.findall(keys_dtype)[0]
            keys_dtype = np.dtype(keys_dtype)

        _argsorted_unknown_idx = attrs.pop('_argsorted_unknown_idx', None)

        loaded = NominalDataEncoder(
            np.array(h5['keys'], dtype=keys_dtype),
            ignore_dups=False,
            sort_keys=False,
            **attrs,
        )

        if '_argsorted_keys' in h5:
            loaded._argsorted_keys = np.array(h5['_argsorted_keys'])
            loaded._argsorted_unknown_idx = _argsorted_unknown_idx

        if close:
            h5.close()
        return loaded


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
