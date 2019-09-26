from .images import Images
import imageio
import typing


class Video(Images):

    def __init__(self,
                 file_path: str = None,
                 channel_mode: str = 'RGB',
                 debug_mode: bool = False) -> None:

        super(Video, self).__init__(images=None,
                                    channel_mode=channel_mode,
                                    debug_mode=debug_mode)
        self.load(file_path)

    def load(self, file_path: typing.Optional[str] = None) -> None:
        if file_path is not None:
            reader = imageio.get_reader(file_path, "ffmpeg")
            self.images = [image for image in reader]

    def save(self):
        raise NotImplementedError
