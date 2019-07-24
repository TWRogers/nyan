import numpy as np


COLOUR_MODES = ('RGB', 'BGR')


class Collection(object):

    def __init__(self,
                 images: list = None,
                 channel_mode: str = 'RGB',
                 debug_mode: bool = False) -> None:

        self.channel_mode = channel_mode
        self.debug_mode = debug_mode
        if images is None:
            self.images = []
        else:
            self.images = images

    @classmethod
    def from_array(cls,
                   array: np.ndarray,
                   channel_mode: str = 'RGB',
                   debug_mode: bool = False):

        if array.ndim == 4:
            return cls(images=[x for x in array],
                       channel_mode=channel_mode,
                       debug_mode=debug_mode)

        elif array.ndim == 3:
            if array.shape[2] == 1:
                assert channel_mode is None, 'channel_mode must be None if array.ndim == 3 and array.shape[2] == 1'
                return cls(images=[array],
                           channel_mode=channel_mode,
                           debug_mode=debug_mode)
            elif array.shape[2] == 3:
                assert channel_mode in COLOUR_MODES, \
                    'channel_mode must be in {} if array.ndim == 3 and array.shape[2] == 3'.format(COLOUR_MODES)
                return cls(images=[array],
                           channel_mode=channel_mode,
                           debug_mode=debug_mode)

        elif array.ndim == 2:
            assert channel_mode is None, 'channel_mode must be None if array.ndim == 2'
            return cls(images=[np.expand_dims(array, axis=2)],
                       channel_mode=channel_mode,
                       debug_mode=debug_mode)

    @property
    def __array_interface__(self) -> dict:
        # support for casting to a numpy array
        array = self.as_array()
        return {"typestr": array.dtype,
                "shape": self.shape,
                "version": 3,
                "data": array.tobytes()}

    # @property
    # def __array_interface__(self):
    #     # numpy array interface support
    #     new = {}
    #     shape, typestr = _conv_type_shape(self)
    #     new["shape"] = shape
    #     new["typestr"] = typestr
    #     new["version"] = 3
    #     if self.mode == "1":
    #         # Binary images need to be extended from bits to bytes
    #         # See: https://github.com/python-pillow/Pillow/issues/350
    #         new["data"] = self.tobytes("raw", "L")
    #     else:
    #         new["data"] = self.tobytes()
    #     return new

    def as_array(self) -> np.ndarray:
        assert self.size is not None, 'All images must have same size to export as array.'
        return np.array(self.images)

    @property
    def shape(self):
        if self.images:
            shapes = list(set(map(lambda x: x.shape, self.images)))
            if len(shapes) == 1:
                return shapes[0]
            else:
                return None
        else:
            return None

    @property
    def size(self):
        if self.images:
            sizes = list(set(map(lambda x: x.shape[:2][::-1], self.images)))
            if len(sizes) == 1:
                return sizes[0]
            else:
                return None
        else:
            return None

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return "<{module}.{name} images channel_mode={channel_mode} size={size} at 0x{id}>".format(
            module=self.__class__.__module__,
            name=self.__class__.__name__,
            channel_mode=self.channel_mode,
            size=self.size,
            id=id(self))
