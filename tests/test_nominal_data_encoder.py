"""Tests of the compiler-like checking of docstrings to ensure they parse
styles as they are expected and includes examples of out-of-style
docstrings.

The fixtures are the expected parsed tokens from docstr.parse.
"""
from collections import OrderedDict
from copy import copy, deepcopy
import os
from string import ascii_uppercase, ascii_lowercase

from bidict import OrderedBidict
import pytest
import numpy as np

from exputils.data.labels import NominalDataEncoder as NDE


@pytest.fixture
def example_labels():
    return tuple(
        [str(i) for i in range(10)]
        + list(ascii_uppercase)
        + list(ascii_lowercase)
        + ['noxious']
    )


@pytest.fixture
def example_labels_file(request):
    return os.path.join(request.fspath.dirname, 'example_labels.txt')


@pytest.fixture
def example_labels_bidict(example_labels):
    return OrderedBidict({key: enc for enc, key in enumerate(example_labels)})


@pytest.mark.dependency(name='object_basics', scope='session')
#@pytest.mark.incremental
class TestObjectBasics:
    """The fundamental parsing of functions including defaults and choices."""
    def test_default_init_and_eq(self, example_labels, example_labels_bidict):
        nde_1 = NDE(example_labels)

        argsorted = np.unique(example_labels, return_index=True)[1]

        # Test the argsorted_keys given these example labels
        assert (nde_1.argsorted_keys == argsorted).all()

        # Test the default attributes
        assert nde_1.pos_label == 1
        assert nde_1.neg_label == 0
        assert nde_1.sparse_output is False
        assert nde_1._unknown_key is None

        # Check the order of example labels in the encoder, and check list conv
        assert tuple(nde_1.encoder) == example_labels
        assert tuple(nde_1) == example_labels
        assert nde_1.encoder == example_labels_bidict

        # Test the default properties
        assert nde_1.unknown_key is None
        assert nde_1.unknown_idx is None
        assert nde_1.shift is 0

        # Test equality check `==`, `!=`, `is`, `is not` between NDEs.
        nde_2 = NDE(example_labels)

        # Repeat tests on nde_2 as sanity check for any global changes in init
        # Test the argsorted_keys given these example labels
        assert (nde_2.argsorted_keys == argsorted).all()

        # Test the default attributes
        assert nde_2.pos_label == 1
        assert nde_2.neg_label == 0
        assert nde_2.sparse_output is False
        assert nde_2._unknown_key is None

        # Check the order of example labels in the encoder, and check list conv
        assert tuple(nde_2.encoder) == example_labels
        assert tuple(nde_2) == example_labels
        assert nde_2.encoder == example_labels_bidict

        # Test the default properties
        assert nde_2.unknown_key is None
        assert nde_2.unknown_idx is None
        assert nde_2.shift is 0

        # Test comparison of NDEs
        assert nde_1 == nde_2
        assert not (nde_1 != nde_2)
        assert not (nde_1 is nde_2)
        assert nde_1 is not nde_2

        # TODO test error on ignore_dups = False
        # TODO test no error on ignore_dups = True

    def test_load_save(self, example_labels, example_labels_file):
        assert NDE.load(example_labels_file) == NDE(example_labels)

    def test_copy(self, example_labels):
        nde_src = NDE(example_labels)
        nde_copy = copy(nde_src)
        assert nde_src == nde_copy

        # Change an object within one, witness change in the other
        nde_src.argsorted_keys[0] = -100
        assert nde_src == nde_copy

        nde_copy.argsorted_keys[-1] = -200
        assert nde_src == nde_copy

        # Change a literal within one, witness no change in the other
        tmp = nde_src.pos_label
        nde_src.pos_label = 100
        assert nde_src != nde_copy
        nde_src.pos_label = tmp

        nde_copy.pos_label = 100
        assert nde_src != nde_copy

    def test_deepcopy(self, example_labels):
        nde_src = NDE(example_labels)
        nde_copy = deepcopy(nde_src)
        assert nde_src == nde_copy

        # Change an object within one, witness no change in the other
        tmp = nde_src.argsorted_keys[0]
        nde_src.argsorted_keys[0] = -100
        assert nde_src != nde_copy
        nde_src.argsorted_keys[0] = tmp
        assert nde_src == nde_copy

        tmp = nde_copy.argsorted_keys[-1]
        nde_copy.argsorted_keys[-1] = -200
        assert nde_src != nde_copy

    def test_encoder_bidict_methods(self, example_labels):
        """Methods that are pass throughs to the bidict encoder."""
        nde = NDE(example_labels)

        # Double Unders
        # Test len
        assert len(nde) == len(nde.encoder)

        # Test contains, getitem, dict method: get
        for i, label in enumerate(example_labels):
            assert label in nde
            assert nde[label] == i
            assert nde.get(label) == i

        # Test iter and reversed
        assert tuple(iter(nde)) == example_labels
        assert tuple(reversed(nde)) == tuple(reversed(example_labels))

        # Dict methods: keys, values, items, get
        assert nde.keys() == nde.encoder.keys()
        assert nde.values() == nde.encoder.values()
        assert nde.items() == nde.encoder.items()
        assert nde.get('pie', None) is None

        # Bidict properties / methods
        assert nde.inverse is nde.encoder.inverse
        assert nde.inv is nde.encoder.inv
        assert nde.inv is nde.encoder.inverse


@pytest.mark.dependency(name='encoder_knowns', depends=['object_basics'])
class TestLabelEncoder:
    @pytest.mark.parametrize(
        'one_hot,shift,unknown_key',
        [
            (False, 0, None),
            (True, 0, None),
            (False, 20, None),
            (True, 20, None),
            # label '0' treated as unknown
            (False, 0, '0'),
            (True, 0, '0'),
            (False, 20, '0'),
            (True, 20, '0'),
            # Unknown added at first idx
            (False, 0, 'unknown'),
            (True, 0, 'unknown'),
            (False, 20, 'unknown'),
            (True, 20, 'unknown'),
        ],
    )
    def test_encode_decode(self, one_hot, shift, unknown_key, example_labels):
        nde = NDE(example_labels, shift=shift, unknown_key=unknown_key)

        if unknown_key not in {None, '0'}:
            example_labels = tuple(['unknown'] + list(example_labels))

        ref_labels = np.array(example_labels)
        n_labels = len(example_labels)

        if one_hot:
            ref_enc = np.eye(n_labels)
            decode_axes = [-1] * 10
            decode_axes_x16 = [-1] * 4
        else:
            ref_enc = np.arange(n_labels) + shift
            decode_axes = [None] * 10
            decode_axes_x16 = [None] * 4

        # Check encoding of a list and tuple
        assert nde.shift == shift
        assert (nde.encode(example_labels, one_hot) == ref_enc).all()
        assert (nde.encode(list(example_labels), one_hot) == ref_enc).all()

        # Check np.ndarray encoding and decoding of different shapes
        if unknown_key not in {None, '0'}:
            shapes = (
                [n_labels],
                [1, n_labels],
                [1, 1, 1, n_labels, 1],
                [8, 8],
                [32, 2],
                [2, 16, 2],
                [4, 16],
                [16, 4],
                [2, 2, 4, 4],
                [2, 2, 2, 2, 4],
            )
        else:
            shapes = (
                [n_labels],
                [1, n_labels],
                [1, 1, 1, n_labels, 1],
                [3, 21],
                [21, 3],
                [7, 3, 3],
                [3, 7, 3],
                [3, 3, 7],
                [9, 7],
                [7, 9],
            )
        for i, shape in enumerate(shapes):
            encoded = nde.encode(ref_labels.reshape(shape), one_hot)
            enc_shape = shape + [n_labels] if one_hot else shape
            assert (encoded == ref_enc.reshape(enc_shape)).all()
            assert (
                nde.decode(encoded, decode_axes[i])
                == ref_labels.reshape(shape)
            ).all()

        if unknown_key != '0':
            return

        # Check np.ndarray encoding and decoding of larger dimensions
        ref_labels_x16 = np.concatenate([ref_labels] * 16)
        ref_enc_x16 = np.concatenate([ref_enc] * 16)
        shapes_x16 = (
            [2, 8, 3, 7, 3],
            [4, 4, 3, 7, 3],
            [2, 2, 2, 2, 3, 7, 3],
            [1, 16, 3, 7, 3],
        )
        for i, shape in enumerate(shapes_x16):
            encoded = nde.encode(ref_labels_x16.reshape(shape), one_hot)
            enc_shape = shape + [n_labels] if one_hot else shape
            assert (encoded == ref_enc_x16.reshape(enc_shape)).all()
            assert (
                nde.decode(encoded, decode_axes_x16[i])
                == ref_labels_x16.reshape(shape)
            ).all()

    def test_shift_encoding(self, example_labels):
        """Tests the shift_encoding method for positive and negative shifts."""
        nde = NDE(example_labels)
        assert nde.shift == 0

        nde_shifted = NDE(example_labels, shift=20)
        assert nde_shifted.shift == 20

        nde.shift_encoding(20)

        assert nde == nde_shifted

        nde = NDE(example_labels)
        nde_shifted.shift_encoding(-20)
        assert nde == nde_shifted

    def test_append_pop(self, example_labels):
        """Tests the append and pop methods."""
        nde = NDE(example_labels)
        nde_copy = deepcopy(nde)
        nde_appended = NDE(list(example_labels) + ['pie'])

        assert nde != nde_appended
        assert nde_copy != nde_appended
        assert nde == nde_copy

        # Test an individual key first.
        nde.append('pie')

        assert nde != nde_copy
        assert nde == nde_appended

        nde.pop('pie')

        assert nde == nde_copy
        assert nde != nde_appended

        # Test multiple keyes
        nde_appended = NDE(list(example_labels) + ['pie', 'cake', 'cheese'])
        nde.append(['pie', 'cake', 'cheese'])
        assert nde == nde_appended
        assert nde != nde_copy

        nde_appended_copy = deepcopy(nde)
        assert nde == nde_appended_copy

        # TODO Include pop multiple keys? nde.pop(['pie', 'cake', 'cheese'])
        for key in ['pie', 'cake', 'cheese']:
            nde.pop(key)
        assert nde == nde_copy

        for key in ['pie', 'cake', 'cheese']:
            nde.append(key)
        assert nde == nde_appended

        # Test removal of a key value pair using the encoding.
        for i in range(1, 4):
            nde.pop(66 - i, True)
        assert nde == nde_copy

        # Test removal of a key in the middle
        ex_labels_miss_A = list(example_labels)
        del ex_labels_miss_A[10]
        nde_missing_A = NDE(ex_labels_miss_A)

        nde.pop('A')
        nde == nde_missing_A

        # Test shift update after pop
        nde = NDE(example_labels, shift=12)
        assert nde.shift == 12
        nde.pop('0')
        assert nde.shift == 12

    #@pytest.mark.xfail
    #TODO def test_reorder(self, example_labels):
    #    assert False


#@pytest.mark.dependency(depends=['encoder_knowns'])
class TestUnknownLabel:
    def test_init_unknown_key_given_in_ordered_keys(self, example_labels):
        # At beginning
        nde = NDE(['unknown'] + list(example_labels), unknown_key='unknown')
        assert nde.unknown_key == 'unknown'
        assert nde.unknown_idx == 0

        nde = NDE(
            ['unknown'] + list(example_labels),
            unknown_key='unknown',
            unknown_idx=0,
        )
        assert nde.unknown_key == 'unknown'
        assert nde.unknown_idx == 0

        # At end
        nde = NDE(list(example_labels) + ['unknown'], unknown_key='unknown')
        assert nde.unknown_key == 'unknown'
        assert nde.unknown_idx == len(example_labels)

        nde = NDE(
            list(example_labels) + ['unknown'],
            unknown_key='unknown',
            unknown_idx=len(example_labels),
        )
        assert nde.unknown_key == 'unknown'
        assert nde.unknown_idx == len(example_labels)

        # In middle
        nde = NDE(example_labels, unknown_key='A')
        assert nde.unknown_key == 'A'
        assert nde.unknown_idx == 10

        # Expects ValueError
        raised_error = False
        try:
            nde = NDE(example_labels, unknown_key='A', unknown_idx=22)
        except ValueError:
            raised_error = True
        assert raised_error

    def test_init_unknown_key_given_not_in_ordered_keys(self, example_labels):
        # At beginning
        nde = NDE(example_labels, unknown_key='unknown')
        assert nde.unknown_key == 'unknown'
        assert nde.unknown_idx == 0

        # At end
        nde = NDE(example_labels, unknown_key='unknown', unknown_idx=-1)
        assert nde.unknown_key == 'unknown'
        assert nde.unknown_idx == len(example_labels)

        # In middle
        assert nde['A'] == 10
        nde = NDE(example_labels, unknown_key='unknown', unknown_idx=10)
        assert nde.unknown_key == 'unknown'
        assert nde.unknown_idx == 10
        assert nde['A'] == 11

        # Expects ValueError
        raised_error = False
        try:
            nde = NDE(example_labels, unknown_key='unknown', unknown_idx=220)
        except TypeError:
            raised_error = True
        assert raised_error

    def test_shift(self, example_labels):
        nde = NDE(example_labels, unknown_key='unknown')
        assert nde.unknown_key == 'unknown'
        assert nde.unknown_idx == 0
        assert nde.shift == 0

        nde_copy = deepcopy(nde)
        assert nde == nde_copy

        nde_shifted = NDE(example_labels, shift=20, unknown_key='unknown')
        assert nde_shifted.shift == 20
        assert nde_shifted.unknown_key == 'unknown'
        assert nde_shifted.unknown_idx == 20

        nde.shift_encoding(20)

        assert nde == nde_shifted

        nde_shifted.shift_encoding(-20)
        assert nde_copy == nde_shifted

    def test_pop(self, example_labels):
        nde = NDE(example_labels, unknown_key='unknown')
        nde_knowns = NDE(example_labels)
        assert nde.unknown_key == 'unknown'
        assert nde.unknown_idx == 0

        nde.pop(nde.unknown_key)
        assert nde.unknown_key is None
        assert nde.unknown_idx is None
        assert nde == nde_knowns

    def test_pop_shift(self, example_labels):
        nde = NDE(example_labels, unknown_key='unknown', shift=12)
        nde_knowns = NDE(example_labels, shift=12)
        assert nde.unknown_key == 'unknown'
        assert nde.unknown_idx == 12

        assert nde.pop(nde.unknown_key) == 12
        assert nde.unknown_key is None
        assert nde.unknown_idx is None
        assert nde == nde_knowns
        assert nde.shift == 12

    #@pytest.mark.xfail
    #def test_reorder(self):
    #    assert False
