import pytest
from nyan.collections import Video
import os

TEST_VIDEO = os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-3]), 'static/nyan.mp4')


@pytest.mark.parametrize('debug_mode', [True, False])
def test_video_init(debug_mode):
    video = Video(fp=TEST_VIDEO, debug_mode=debug_mode)
    assert video.size == (720, 480)
    assert video.channel_mode == ('R', 'G', 'B')
