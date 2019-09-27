import pytest
from nyan.collections import Images
import numpy as np


@pytest.mark.parametrize('debug_mode', [True, False])
@pytest.mark.parametrize('c_channels', [1, 3])
def test_collection_size(debug_mode, c_channels):
    if c_channels == 1:
        array = np.zeros((100, 100, 1))
        collection = Images.from_array(array, channel_mode=('GRAY',), debug_mode=debug_mode)
        assert collection.size == (100, 100)

        array = np.zeros((10, 100, 100, 1))
        collection = Images.from_array(array, channel_mode=('GRAY',), debug_mode=debug_mode)
        assert collection.size == (100, 100)

    elif c_channels == 3:
        array = np.zeros((100, 100, 3))
        collection = Images.from_array(array,  channel_mode=('R', 'G', 'B'), debug_mode=debug_mode)
        assert collection.size == (100, 100)

        array = np.zeros((10, 100, 100, 3))
        collection = Images.from_array(array,  channel_mode=('R', 'G', 'B'), debug_mode=debug_mode)
        assert collection.size == (100, 100)
    else:
        raise ValueError


@pytest.mark.parametrize('debug_mode', [True, False])
@pytest.mark.parametrize('c_channels', [1, 3])
def test_collection_len(debug_mode, c_channels):
    if c_channels == 1:
        array = np.zeros((100, 100, 1))
        collection = Images.from_array(array, channel_mode=('GRAY',), debug_mode=debug_mode)
        assert len(collection) == 1

        array = np.zeros((10, 100, 100, 1))
        collection = Images.from_array(array,  channel_mode=('GRAY',), debug_mode=debug_mode)
        assert len(collection) == 10

    elif c_channels == 3:
        array = np.zeros((100, 100, 3))
        collection = Images.from_array(array,  channel_mode=('R', 'G', 'B'), debug_mode=debug_mode)
        assert len(collection) == 1

        array = np.zeros((10, 100, 100, 3))
        collection = Images.from_array(array,  channel_mode=('R', 'G', 'B'), debug_mode=debug_mode)
        assert len(collection) == 10
    else:
        raise ValueError


@pytest.mark.parametrize('debug_mode', [True, False])
@pytest.mark.parametrize('c_channels', [1, 3])
def test_collection_as_array(debug_mode, c_channels):
    if c_channels == 1:
        array = np.zeros((10, 100, 100, 1))
    elif c_channels == 3:
        array = np.zeros((10, 100, 100, 3))
    else:
        raise ValueError

    collection = Images.from_array(array, debug_mode=debug_mode)
    assert np.all(collection.as_array() == array)


@pytest.mark.parametrize('debug_mode', [True, False])
@pytest.mark.parametrize('c_channels', [1, 3])
def test_collection_copy(debug_mode, c_channels):
    if c_channels == 1:
        array = np.zeros((10, 100, 100, 1))
        channel_mode = ('GRAY',)
    elif c_channels == 3:
        array = np.zeros((10, 100, 100, 3))
        channel_mode = ('R', 'G', 'B')
    else:
        raise ValueError

    collection = Images.from_array(array, channel_mode=channel_mode, debug_mode=debug_mode)
    collection_copy = collection.copy()
    assert collection_copy == collection
