

class Collection(object):

    def __init__(self) -> None:
        self.foo = 1

    @classmethod
    def from_array(cls):
        raise NotImplementedError

    @property
    def __array_interface__(self) -> dict:
        # support for casting to a numpy array
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> foo={self.foo}"
