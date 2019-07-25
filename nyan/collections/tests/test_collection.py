import pytest
from nyan.collections import Collection, COLOUR_MODES
import numpy as np


@pytest.mark.parametrize('debug_mode', [True, False])
@pytest.mark.parametrize('channel_mode', ['RGB', 'BGR', None])
def test_collection_size(debug_mode, channel_mode):
    if channel_mode is None:
        array = np.zeros((100, 100, 1))
        collection = Collection.from_array(array, channel_mode=channel_mode, debug_mode=debug_mode)
        assert collection.size == (100, 100)

        array = np.zeros((10, 100, 100, 1))
        collection = Collection.from_array(array, channel_mode=channel_mode, debug_mode=debug_mode)
        assert collection.size == (100, 100)

    elif channel_mode in COLOUR_MODES:
        array = np.zeros((100, 100, 3))
        collection = Collection.from_array(array, channel_mode=channel_mode, debug_mode=debug_mode)
        assert collection.size == (100, 100)

        array = np.zeros((10, 100, 100, 3))
        collection = Collection.from_array(array, channel_mode=channel_mode, debug_mode=debug_mode)
        assert collection.size == (100, 100)
    else:
        raise ValueError


@pytest.mark.parametrize('debug_mode', [True, False])
@pytest.mark.parametrize('channel_mode', ['RGB', 'BGR', None])
def test_collection_len(debug_mode, channel_mode):
    if channel_mode is None:
        array = np.zeros((100, 100, 1))
        collection = Collection.from_array(array, channel_mode=channel_mode, debug_mode=debug_mode)
        assert len(collection) == 1

        array = np.zeros((10, 100, 100, 1))
        collection = Collection.from_array(array, channel_mode=channel_mode, debug_mode=debug_mode)
        assert len(collection) == 10

    elif channel_mode in COLOUR_MODES:
        array = np.zeros((100, 100, 3))
        collection = Collection.from_array(array, channel_mode=channel_mode, debug_mode=debug_mode)
        assert len(collection) == 1

        array = np.zeros((10, 100, 100, 3))
        collection = Collection.from_array(array, channel_mode=channel_mode, debug_mode=debug_mode)
        assert len(collection) == 10
    else:
        raise ValueError


@pytest.mark.parametrize('debug_mode', [True, False])
@pytest.mark.parametrize('channel_mode', ['RGB', 'BGR', None])
def test_collection_as_array(debug_mode, channel_mode):
    if channel_mode is None:
        array = np.zeros((10, 100, 100, 1))
    elif channel_mode in COLOUR_MODES:
        array = np.zeros((10, 100, 100, 3))
    else:
        raise ValueError

    collection = Collection.from_array(array, channel_mode=channel_mode, debug_mode=debug_mode)
    assert np.all(collection.as_array() == array)


# @pytest.mark.parametrize('debug_mode', [True, False])
# @pytest.mark.parametrize('channel_mode', ['RGB', 'BGR', None])
# def test_collection_array_interface(debug_mode, channel_mode):
#     if channel_mode is None:
#         array = np.zeros((10, 100, 100, 1))
#     elif channel_mode in COLOUR_MODES:
#         array = np.zeros((10, 100, 100, 3))
#     else:
#         raise ValueError
#
#     collection = Collection.from_array(array, channel_mode=channel_mode, debug_mode=debug_mode)
#     assert np.all(np.array(collection) == array)
