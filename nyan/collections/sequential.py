from nyan.utils import IMREAD_FN
from .images import Images
import glob
import typing


class Sequential(Images):

    def __init__(self,
                 wildcard: typing.Optional[str] = None,
                 channel_mode: str = 'RGB',
                 debug_mode: bool = False) -> None:

        super(Sequential, self).__init__(images=None,
                                         channel_mode=channel_mode,
                                         debug_mode=debug_mode)
        self.load(wildcard)

    def load(self, wildcard: typing.Optional[str] = None) -> None:
        self.images = [IMREAD_FN(file_path) for file_path in glob.glob(wildcard)]