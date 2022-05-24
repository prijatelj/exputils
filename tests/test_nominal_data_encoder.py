"""Tests of the compiler-like checking of docstrings to ensure they parse
styles as they are expected and includes examples of out-of-style
docstrings.

The fixtures are the expected parsed tokens from docstr.parse.
"""
from collections import OrderedDict
import copy

import pytest

from exputils.data.labels import NominalDataEncoder


@pytest.fixture
def abc_labels():
    return [
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'g',
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        '0',
        '1',
        '2',
        '3',
        'noxious',
    ]

@pytest.mark.dependency(name='object_basics', scope='session')
#@pytest.mark.incremental
@pytest.mark.xfail
class TestObjectBasics:
    """The fundamental parsing of functions including defaults and choices."""
    def test_default_init(self):
        assert False

    def test_load_save(self, abc_labels):
        assert False

    def test_copy(self):
        assert False

    def test_deepcopy(self):
        assert False

    def test_encoder_bidict_methods(self):
        """Methods that are pass throughs to the bidict encoder."""
        assert False
        # TODO Double Unders
        # len
        # contains
        # getitem
        # iter
        # reverse

        # TODO Dict methods
        # keys
        # values
        # items
        # get

        # TODO Bidict properties / methods
        # inverse
        # inv


@pytest.mark.dependency(name='encoder_knowns', depends=['object_basics'])
@pytest.mark.xfail
class TestLabelEncoder:
    def test_encode_decode(self):
        assert False

    def test_shift(self):
        assert False

    def test_pop(self):
        assert False

    def test_reorder(self):
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
