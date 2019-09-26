from .images import Images
import imageio
import typing
import numpy as np


class Image(Images):

    def __init__(self,
                 filepath: typing.Optional[str] = None,
                 channel_mode: tuple = ('R', 'G', 'B'),
                 debug_mode: bool = False) -> None:

        super(Image, self).__init__(images=None,
                                    channel_mode=channel_mode,
                                    debug_mode=debug_mode)
        self.load(filepath)

    def load(self, filepath: typing.Optional[str] = None) -> None:
        if filepath is not None:
            reader = imageio.get_reader(filepath, 'ffmpeg')
            self.images = [image for image in reader]

    @property
    def image(self):
        return self.images[0]

    def as_array(self) -> np.ndarray:
        return super(Image, self).as_array()[0, ...]

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
