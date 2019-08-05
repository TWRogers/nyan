import pytest
from nyan.collections import Video
import os

TEST_VIDEO = os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-3]), 'static/nyan.mp4')


@pytest.mark.parametrize('debug_mode', [True, False])
@pytest.mark.parametrize('channel_mode', ['RGB', 'BGR', None])
def test_video_init(debug_mode, channel_mode):
    video = Video(filepath=TEST_VIDEO, channel_mode=channel_mode, debug_mode=debug_mode)
    assert video.size == (720, 480)
