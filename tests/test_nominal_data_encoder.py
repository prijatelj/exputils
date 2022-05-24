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

        # Test comparison of NDEs
        assert nde_1 == nde_2
        assert not (nde_1 != nde_2)
        assert not (nde_1 is nde_2)
        assert nde_1 is not nde_2

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
    @pytest.mark.parametrize('one_hot', [False, True])
    def test_encode_decode(self, one_hot, example_labels):
        n_labels = len(example_labels)
        ref_labels = np.array(example_labels)
        if one_hot:
            ref_enc = np.eye(n_labels)
            decode_axes = [-1] * 10
            decode_axes_x16 = [-1] * 4
        else:
            ref_enc = np.arange(n_labels)
            decode_axes = [None] * 10
            decode_axes_x16 = [None] * 4

        # Check encoding of a list and tuple
        nde = NDE(example_labels)
        assert (nde.encode(example_labels, one_hot) == ref_enc).all()
        assert (nde.encode(list(example_labels), one_hot) == ref_enc).all()

        # Check np.ndarray encoding and decoding of different shapes
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

    @pytest.mark.xfail
    def test_shift(self, example_labels):
        assert False

    @pytest.mark.xfail
    def test_pop(self, example_labels):
        assert False

    @pytest.mark.xfail
    def test_reorder(self, example_labels):
        assert False


@pytest.mark.dependency(depends=['encoder_knowns'])
@pytest.mark.xfail
class TestUnknownLabel:
    def test_init_unknown_key_given_in_ordered_keys(self):
        assert False

    def test_init_unknown_key_given_not_in_ordered_keys(self):
        assert False

    # Test label encoder with unknown label
    def test_encode_decode(self):
        assert False

    def test_shift(self):
        assert False

    def test_pop(self):
        assert False

    def test_reorder(self):
        assert False
