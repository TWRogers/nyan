from .images import Images
from nyan.utils import IMREAD_FN
import numpy as np


class Image(Images):
    def __init__(self, src: str, debug_mode: bool = False) -> None:

        super(Image, self).__init__(src=src, debug_mode=debug_mode)

    def _load(self, fp: str) -> None:
        self.images = [IMREAD_FN(fp)]

    @property
    def image(self):
        return self.images[0]

    def as_array(self) -> np.ndarray:
        return super(Image, self).as_array()[0, ...]

    def save(self):
        raise NotImplementedError
