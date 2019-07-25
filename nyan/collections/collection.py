import numpy as np


COLOUR_MODES = ('RGB', 'BGR')


class Collection(object):

    def __init__(self,
                 images: list = None,
                 channel_mode: str = 'RGB',
                 debug_mode: bool = False) -> None:

        self.channel_mode = channel_mode
        self.debug_mode = debug_mode

        self.images = [] if images is None else images
        self._original_images = [] if images is None else images.copy()

        self._transform_history = {}
        self.debug_history = {}

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

    def as_array(self) -> np.ndarray:
        assert self.size is not None, 'All images must have same size to export as array.'
        return np.array(self.images)

    @property
    def shape(self) -> tuple:
        if self.images:
            shapes = list(set(map(lambda x: x.shape, self.images)))
            if len(shapes) == 1:
                return shapes[0]
            else:
                return None
        else:
            return None

    @property
    def size(self) -> tuple:
        if self.images:
            sizes = list(set(map(lambda x: x.shape[:2][::-1], self.images)))
            if len(sizes) == 1:
                return sizes[0]
            else:
                return None
        else:
            return None

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is None:
                step = 1
            else:
                step = key.step
            return [self.images[i] for i in range(key.start, key.stop, step)]

        elif isinstance(key, tuple):
            if isinstance(key[0], slice):
                if key[0].step is None:
                    step = 1
                else:
                    step = key[0].step
                return [self.images[i][key[1:]] for i in range(key[0].start, key[0].stop, step)]
            else:
                return [self.images[key[0]][key[1:]]]

        return self.images[key]

    def __iter__(self) -> iter:
        return iter(self.images)

    def __eq__(self, other_object) -> bool:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.images)

    def __repr__(self) -> str:
        return "<{module}.{name} images channel_mode={channel_mode} size={size} at 0x{id}>".format(
            module=self.__class__.__module__,
            name=self.__class__.__name__,
            channel_mode=self.channel_mode,
            size=self.size,
            id=id(self))

    def copy(self):


        self.channel_mode = channel_mode
        self.debug_mode = debug_mode

        self.images = [] if images is None else images
        self._original_images = [] if images is None else images.copy()

        self._transform_history = {}
        self.debug_history = {}

        raise NotImplemented

    def __add_to_transform_history(self,
                                   crop: tuple = None,
                                   resize: tuple = None,
                                   pad: tuple = None):

        self._transform_history.update({"crop": crop,
                                        "resize": resize,
                                        "pad": pad})

    def transform_image(self, start_event, end_event) -> np.ndarray:
        raise NotImplementedError

    def transform_point(self, start_event, end_event) -> tuple:
        raise NotImplementedError

    def transform_box(self, start_event, end_event) -> tuple:
        raise NotImplementedError

    def transform_polygon(self, vertices, start_event, end_event) -> list:
        raise NotImplementedError

    def crop(self, cropping_params: dict) -> None:
        raise NotImplementedError

    def _crop(self) -> None:
        raise NotImplementedError

    def resize(self,
               target_size: tuple,
               preserve_aspect_ratio: bool = False) -> None:
        raise NotImplementedError

    def _resize(self) -> None:
        raise NotImplementedError

    def pad(self, padding_params) -> None:
        raise NotImplementedError

    def label_event(self) -> None:
        raise NotImplementedError

    @property
    def __array_interface__(self) -> dict:
        raise NotImplementedError
    #     # support for casting to a numpy array
    #     array = self.as_array()
    #     return {"typestr": array.dtype,
    #             "shape": self.shape,
    #             "version": 3,
    #             "data": array.tobytes()}
