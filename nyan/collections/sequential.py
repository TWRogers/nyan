from nyan.utils import IMREAD_FN
from .images import Images
import glob


class Sequential(Images):
    def __init__(self, directory_wildcard: str,
                 debug_mode: bool = False) -> None:

        super(Sequential, self).__init__(src=directory_wildcard,
                                         debug_mode=debug_mode)
        self.load(directory_wildcard)

    def _load(self, fp: str) -> None:
        self.images = [IMREAD_FN(file_path) for file_path in glob.glob(fp)]

    def save(self):
        raise NotImplementedError
