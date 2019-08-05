from .images import Images
import imageio
import typing


class Video(Images):

    def __init__(self,
                 filepath: typing.Optional[str] = None,
                 channel_mode: str = 'RGB',
                 debug_mode: bool = False) -> None:

        super(Video, self).__init__(images=None,
                                    channel_mode=channel_mode,
                                    debug_mode=debug_mode)
        self.load(filepath)

    def load(self, filepath: typing.Optional[str] = None) -> None:
        if filepath is not None:
            reader = imageio.get_reader(filepath, 'ffmpeg')
            self.images = [image for image in reader]
