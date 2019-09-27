from .images import Images
import imageio


class Video(Images):
    def __init__(self, fp: str = None, debug_mode: bool = False) -> None:

        super(Video, self).__init__(src=fp, debug_mode=debug_mode)

    def _load(self, fp: str) -> None:
        reader = imageio.get_reader(fp, "ffmpeg")
        self.images = [image for image in reader]

    def save(self):
        raise NotImplementedError
