import typing
import numpy as np
from nyan.utils import History
import cv2
import matplotlib.pyplot as plt
import math

COLOUR_MODES = ('RGB', 'BGR')


class Images(object):

    def __init__(self,
                 images: typing.Optional[list] = None,
                 channel_mode: str = 'RGB',
                 debug_mode: bool = False) -> None:

        self.channel_mode = channel_mode
        self.debug_mode = debug_mode

        self.images = [] if images is None else images
        self._original_images = [] if images is None else images.copy()

        self._transform_history = []
        self._debug_history = []

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    @classmethod
    def from_array(cls,
                   array: np.ndarray,
                   channel_mode: typing.Optional[str] = 'RGB',
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
    def shape(self) -> tuple or None:
        if self.images:
            shapes = list(set(map(lambda x: x.shape, self.images)))
            if len(shapes) == 1:
                return shapes[0]
            else:
                return None
        else:
            return None

    @property
    def size(self) -> tuple or None:
        if self.images:
            sizes = list(set(map(lambda x: x.shape[:2][::-1], self.images)))
            if len(sizes) == 1:
                return sizes[0]
            else:
                return None
        else:
            return None

    def __getitem__(self, key):
        new_obj = self.copy()
        if isinstance(key, slice):
            new_obj.images = [self.images[i] for i in range(key.start, key.stop, key.step if key.step else 1)]
            return new_obj

        elif isinstance(key, tuple):
            if isinstance(key[0], slice):
                return [self.images[i][key[1:]] for i in
                        range(key[0].start, key[0].stop, key[0].step if key[0].step else 1)]
            else:
                return [self.images[key[0]][key[1:]]]

        return self.images[key]

    def __iter__(self) -> iter:
        return iter(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        return "<{module}.{name} {n_images} images channel_mode={channel_mode} size={size} at 0x{id}>".format(
            module=self.__class__.__module__,
            name=self.__class__.__name__,
            n_images=len(self),
            channel_mode=self.channel_mode,
            size=self.size,
            id=id(self))

    def copy(self):

        new_copy = self.__class__()

        new_copy.channel_mode = self.channel_mode
        new_copy.debug_mode = self.debug_mode
        new_copy.images = [im.copy() for im in self.images]
        new_copy._original_images = [im.copy() for im in self._original_images]
        new_copy._transform_history = self._transform_history.copy()
        new_copy._debug_history = self._debug_history.copy()

        return new_copy

    def _get_transform_config(self) -> list:
        return [{event['fn_name']: event['fn_args']} for event in self._transform_history]

    def transform_image(self, image, start_event: str = 'start', end_event: str = 'end') -> np.ndarray:
        relevant_transform_history, direction = self._get_relevant_history(start_event, end_event)

        if direction:
            for event in relevant_transform_history:
                getattr(image, event['fn_name'])(*event['fn_args']['args'],
                                                 **event['fn_args']['kwargs'])
        else:
            for event in relevant_transform_history[::-1]:
                try:
                    getattr(image, '{}_inverse'.format(event['fn_name']))(*event['fn_args']['args'],
                                                                          **event['fn_args']['kwargs'])
                except AttributeError:
                    raise NotImplementedError('Inverse fn ({}_inverse) '
                                              'for {} is not Implemented'.format(event['fn_name'], event['fn_name']))
        return image

    def transform_point(self, point, start_event: str = 'start', end_event: str = 'end') -> tuple:
        relevant_transform_history, direction = self._get_relevant_history(start_event, end_event)

        if direction:
            for event in relevant_transform_history:
                fn_name = '{}_point'.format(event['fn_name'])
                try:
                    point = getattr(self, fn_name)(point, *event['fn_args']['args'], **event['fn_args']['kwargs'])
                except AttributeError:
                    raise NotImplementedError('The fn ({})'.format(fn_name))
        else:
            for event in relevant_transform_history[::-1]:
                try:
                    fn_name = '{}_point_inverse'.format(event['fn_name'])
                    point = getattr(self, fn_name)(point, *event['fn_args']['args'], **event['fn_args']['kwargs'])

                except AttributeError:
                    raise NotImplementedError('The fn inverse ({})'.format(fn_name))

        return point

    def transform_box(self, box: tuple, start_event: str = 'start', end_event: str = 'end') -> tuple:
        max_point = self.transform_point((box[1], box[3]), start_event, end_event)
        min_point = self.transform_point((box[0], box[2]), start_event, end_event)
        return min_point[0], max_point[0], min_point[1], max_point[1]

    def transform_polygon(self, vertices: list, start_event: str = 'start', end_event: str = 'end') -> list:
        return [self.transform_point(point, start_event, end_event) for point in vertices]

    def _get_relevant_history(self, start_event: str = 'start', end_event: str = 'end'):
        start_idx = self._get_event_index(start_event)
        end_idx = self._get_event_index(end_event)
        if start_idx < end_idx:
            return self._transform_history[start_idx:end_idx], True
        else:
            return self._transform_history[end_idx:start_idx], False

    def _get_event_index(self, event_name: str) -> int:
        if event_name not in ('start', 'end'):
            event_names = [e.get('label') for e in self._transform_history]
            return event_names.index(event_name)
        elif event_name == 'start':
            return 0
        elif event_name == 'end':
            return len(self._transform_history)

    @History.transform()
    def _rotate(self, angle: float):
        self.images = [cv2.rotate(image, angle) for image in self.images]

    def _rotate_inverse(self, angle: float):
        self.images = [cv2.rotate(image, -angle) for image in self.images]

    def _rotate_point(self, point: tuple, angle: float):
        theta = math.radians(angle)
        x, y = point
        return x * math.cos(theta) - y * math.sin(theta), y * math.cos(theta) - x * math.sin(theta)

    def _rotate_point_inverse(self, point: tuple, angle: float):
        return self._rotate_point(point, -angle)

    def translate(self, x: int = 0, y: int = 0, pad_colour: typing.Optional[tuple] = None):

        self.pad(bottom=y if y > 0 else 0,
                 top=-y if y < 0 else 0,
                 left=x if x > 0 else 0,
                 right=-x if x < 0 else 0,
                 colour=pad_colour)

        self._crop(x_min=0 if x > 0 else -x,
                   x_max=self.size[0]-x if x > 0 else self.size[0],
                   y_min=0 if y > 0 else -y,
                   y_max=self.size[1] - y if y > 0 else self.size[1])

    def crop(self,
             y_min: typing.Optional[int] = None,
             y_max: typing.Optional[int] = None,
             x_min: typing.Optional[int] = None,
             x_max: typing.Optional[int] = None):

        left = 0 if x_min is None else x_min
        top = 0 if y_min is None else y_min
        right = 0 if x_max is None else self.size[0]-x_max
        bottom = 0 if y_max is None else self.size[1]-y_max

        self._crop(top, bottom, left, right)

    @History.transform()
    def _crop(self, top: int, bottom: int, left: int, right: int) -> None:
        self.images = [image[top:-bottom, left:-right] for image in self.images]

    def _crop_inverse(self, top: int, bottom: int, left: int, right: int) -> None:
        return self._pad(top, bottom, left, right, colour=(0, 0, 0))

    def _crop_point(self, point: tuple, top: int, bottom: int, left: int, right: int) -> tuple:
        return point[0]-left, point[1]-top

    def _crop_point_inverse(self, point: tuple, top: int, bottom: int, left: int, right: int) -> tuple:
        return self._pad_point(point, top, bottom, left, right)

    def resize(self,
               target_size: tuple,
               preserve_aspect_ratio: bool = False) -> None:
        # TODO: implement preserve_aspect_ratio
        if (target_size[0] is None) and (target_size[1] is None):
            fx = 1.
            fy = 1.
        elif target_size[0] is None:
            fy = target_size[1] / self.size[1]
            if preserve_aspect_ratio:
                fx = fy
            else:
                fx = 1.
        elif target_size[1] is None:
            fx = target_size[0] / self.size[0]
            if preserve_aspect_ratio:
                fy = fx
            else:
                fy = 1.
        else:
            fx = target_size[0] / self.size[0]
            fy = target_size[1] / self.size[1]

        self._resize(fx=fx, fy=fy)

    @History.transform()
    def _resize(self, fx: float, fy: float) -> None:
        self.images = [cv2.resize(image, dsize=None, fx=fx, fy=fy) for image in self.images]

    def _resize_inverse(self, fx: float, fy: float) -> None:
        self._resize(fx=1./fx, fy=1./fy)

    def _resize_point(self, point: tuple, fx: float, fy: float) -> tuple:
        return point[0]*fx, point[1]*fy

    def _resize_point_inverse(self, point: tuple, fx: float, fy: float) -> tuple:
        return self._resize_point(point, fx=1./fx, fy=1./fy)

    def pad(self,
            top: typing.Optional[int] = None,
            bottom: typing.Optional[int] = None,
            left: typing.Optional[int] = None,
            right: typing.Optional[int] = None,
            colour: typing.Optional[tuple] = None) -> None:

        padding = [x if x is not None else 0 for x in (top, bottom, left, right)]
        if colour is None:
            colour = (0, 0, 0)
        self._pad(*padding, colour)

    @History.transform()
    def _pad(self, top, bottom, left, right, colour):
        self.images = [cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=colour)
                       for image in self.images]

    def _pad_inverse(self, top, bottom, left, right, colour):
        return self._crop(top, bottom, left, right)

    def _pad_point(self, point, top, bottom, left, right, colour):
        return point[0] + left, point[1] + top

    def _pad_point_inverse(self, point, top, bottom, left, right, colour):
        return self._crop_point(point, top, bottom, left, right)

    def label_event(self, label: str) -> None:
        self._transform_history[-1].update({'label': label})

    def show(self, index: int = 0):
        plt.imshow(self.images[index])
        plt.show()

    @property
    def __array_interface__(self) -> dict:
        # TODO: implement numpy array interface
        raise NotImplementedError
    #   # support for casting to a numpy array
    #   array = self.as_array()
    #   return {"typestr": array.dtype,
    #           "shape": self.shape,
    #           "version": 3,
    #           "data": array.tobytes()}
