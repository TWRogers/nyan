import typing
import numpy as np
from nyan.utils import History
import cv2
import matplotlib.pyplot as plt


try:
    import scipy
    from scipy import ndimage
except ImportError:
    scipy = None


COLOUR_MODES = [('R', 'G', 'B'),
                ('B', 'G', 'R'),
                ('X', 'Y', 'Z'),
                ('Y', 'Cr', 'Cb'),
                ('L', 'a', 'b'),
                ('L', 'u', 'v'),
                ('H', 'L', 'S'),
                ('H', 'S', 'V'),
                ('GRAY',)]


class Images(object):

    def __init__(self,
                 src,
                 debug_mode: bool = False) -> None:

        self.debug_mode = debug_mode
        self.channel_mode = None
        self.images = []
        self._original_size = None

        self._transform_history = []
        self._debug_history = []
        if src is not None:
            self.load(src)

    def _load(self, src: str):
        raise NotImplementedError

    def load(self, src: str):
        self._load(src)
        self.channel_mode = ('R', 'G', 'B')
        self._original_size = self.size

    def save(self):
        raise NotImplementedError

    def get_relative_coords(self, normalise=False):
        top_right = self.size
        bottom_left = (0, 0)
        relative_top_right = list(self.transform_point(top_right, 'end', 'start'))
        relative_bottom_left = list(self.transform_point(bottom_left, 'end', 'start'))
        if normalise:
            relative_top_right[0] /= self._original_size[0]
            relative_top_right[1] /= self._original_size[1]
            relative_bottom_left[0] /= self._original_size[0]
            relative_bottom_left[1] /= self._original_size[1]
        return tuple(relative_bottom_left), tuple(relative_top_right)

    def convert_color(self, channel_mode: tuple):
        assert channel_mode in COLOUR_MODES, '{} not a valid colour mode.\n' \
                                             'Please use one of {}'.format(channel_mode, COLOUR_MODES)
        assert self.channel_mode != ('GRAY',), 'Cannot convert from gray to a colour space.'
        conversion_str = "COLOR_{}2{}".format(''.join(self.channel_mode), ''.join(channel_mode))
        assert hasattr(cv2, conversion_str), \
            '{} does not appear to be a valid cv2 colour conversion'.format(conversion_str)
        self.images = [cv2.cvtColor(image, getattr(cv2, conversion_str)) for image in self.images]
        self.channel_mode = channel_mode

    def select_channel(self, channel: str = 'R'):
        assert channel in self.channel_mode, 'channel {} not in channel_mode {}, ' \
                                             'you might need to apply convert_color ' \
                                             'first'.format(channel, self.channel_mode)
        channel_index = self.channel_mode.index(channel)
        self.images = [_repeat_c_channels(image[..., channel_index]) for image in self.images]

    @classmethod
    def from_array(cls,
                   array: np.ndarray,
                   channel_mode: tuple = ('R', 'G', 'B'),
                   debug_mode: bool = False):

        obj = cls(src=None, debug_mode=debug_mode)
        if array.ndim == 4:
            images = [_repeat_c_channels(x) for x in array]

        elif array.ndim == 3:
            if array.shape[2] == 1:
                assert channel_mode == ('GRAY',), "channel_mode must be ('GRAY',) " \
                                                  "if array.ndim == 3 and array.shape[2] == 1"

                images = [_repeat_c_channels(array)]
            elif array.shape[2] == 3:
                assert channel_mode in COLOUR_MODES, \
                    'channel_mode must be in {} if array.ndim == 3 and array.shape[2] == 3'.format(COLOUR_MODES)
                images = [array]

        elif array.ndim == 2:
            assert channel_mode == ('GRAY',), "channel_mode must be ('GRAY',) if array.ndim == 2"
            images = [_repeat_c_channels(array)]

        obj.images = images
        obj._original_size = obj.size
        obj.channel_mode = channel_mode

    def as_array(self) -> np.ndarray:
        assert self.size is not None, 'All images must have same size to convert to numpy array.'
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

    def copy(self):

        new_copy = self.__class__()

        new_copy.channel_mode = self.channel_mode
        new_copy.debug_mode = self.debug_mode
        new_copy.images = [im.copy() for im in self.images]
        new_copy._original_size = self._original_size
        new_copy._transform_history = self._transform_history.copy()
        new_copy._debug_history = self._debug_history.copy()

        return new_copy

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

    def shear(self, shear_intensity: float, fill_mode: str = 'constant',
              c_val: float = 0., interpolation_order: int = 1) -> None:
        self._affine_transform(shear=shear_intensity, fill_mode=fill_mode, c_val=c_val, order=interpolation_order)

    def zoom(self, zoom_range: tuple, fill_mode: str = 'constant', c_val: float = 0.,
             interpolation_order: int = 1) -> None:
        zx, zy = zoom_range
        self._affine_transform(zx=zx, zy=zy, fill_mode=fill_mode, c_val=c_val, order=interpolation_order)

    def rotate(self, theta: float, fill_mode: str = 'constant', c_val: float = 0.,
               interpolation_order: int = 1) -> None:
        self._affine_transform(theta=theta, fill_mode=fill_mode, c_val=c_val, order=interpolation_order)

    def translate(self, x: int = 0, y: int = 0, fill_mode: str = 'constant', c_val: float = 0.,
                  interpolation_order: int = 1):
        self._affine_transform(tx=x, ty=y,
                               fill_mode=fill_mode, c_val=c_val,
                               order=interpolation_order)

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

    def resize(self,
               target_size: tuple,
               preserve_aspect_ratio: bool = False) -> None:
        # TODO: implement preserve_aspect_ratio
        if (target_size[0] is None) and (target_size[1] is None):
            fx, fy = 1., 1.
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

    def pad(self,
            top: typing.Optional[int] = None,
            bottom: typing.Optional[int] = None,
            left: typing.Optional[int] = None,
            right: typing.Optional[int] = None,
            colour: typing.Optional[tuple] = None) -> None:

        padding = [x if x is not None else 0 for x in (top, bottom, left, right)]
        if colour is None:
            colour = (0, 0, 0)
        elif type(colour) != tuple:
            colour = 3*(colour,)
        self._pad(*padding, colour)

    def label_event(self, label: str) -> None:
        self._transform_history[-1].update({'label': label})

    def show(self, index: int = 0):
        plt.imshow(self.images[index])
        plt.show()

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

    @property
    def __array_interface__(self) -> dict:
        return self.as_array().__array_interface__

    @History.transform()
    def _crop(self, top: int, bottom: int, left: int, right: int) -> None:
        w, h = self.size
        self.images = [image[top:h-bottom, left:w-right] for image in self.images]

    def _crop_inverse(self, top: int, bottom: int, left: int, right: int) -> None:
        return self._pad(top, bottom, left, right, colour=(0, 0, 0))

    def _crop_point(self, point: tuple, top: int, bottom: int, left: int, right: int) -> tuple:
        return point[0]-left, point[1]-top

    def _crop_point_inverse(self, point: tuple, top: int, bottom: int, left: int, right: int) -> tuple:
        return self._pad_point(point, top, bottom, left, right, colour=None)

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

    @History.transform()
    def _resize(self, fx: float, fy: float) -> None:
        self.images = [cv2.resize(image, dsize=None, fx=fx, fy=fy) for image in self.images]

    def _resize_inverse(self, fx: float, fy: float) -> None:
        self._resize(fx=1./fx, fy=1./fy)

    def _resize_point(self, point: tuple, fx: float, fy: float) -> tuple:
        return point[0]*fx, point[1]*fy

    def _resize_point_inverse(self, point: tuple, fx: float, fy: float) -> tuple:
        return self._resize_point(point, fx=1./fx, fy=1./fy)

    @History.transform()
    def _affine_transform(self, theta: float = 0, tx: float = 0, ty: float = 0, shear: float = 0,
                          zx: float = 1, zy: float = 1, fill_mode: str = 'nearest', c_val: float = 0.,
                          order: int = 1) -> None:

        final_affine_matrix, final_offset = _get_affine_transformation_matrix(self.size, theta, tx, ty, shear, zx, zy)

        self.images = [_apply_affine_transform(image, final_affine_matrix, final_offset, fill_mode, c_val, order)
                       for image in self.images]

    def _affine_transform_inverse(self, theta: float = 0, tx: float = 0, ty: float = 0, shear: float = 0,
                                  zx: float = 1, zy: float = 1, fill_mode: str = 'nearest', c_val: float = 0.,
                                  order: int = 1) -> None:
        theta = -theta
        tx = -tx
        ty = -ty
        shear = -shear
        zx = 1./zx
        zy = 1./zy

        final_affine_matrix, final_offset = _get_affine_transformation_matrix(self.size, theta, tx, ty, shear, zx, zy)
        self.images = [_apply_affine_transform(image, final_affine_matrix, final_offset, fill_mode, c_val, order)
                       for image in self.images]

    def _affine_transform_point(self, point, theta: float = 0, tx: float = 0, ty: float = 0, shear: float = 0,
                                zx: float = 1, zy: float = 1, fill_mode: str = 'nearest',
                                c_val: float = 0., order: int = 1):

        final_affine_matrix, final_offset = _get_affine_transformation_matrix(self.size, theta, tx, ty, shear,
                                                                              zx, zy)
        point_arr = np.array(point)
        return tuple(np.matmul(final_affine_matrix, point_arr) + final_offset)

    def _affine_transform_point_inverse(self, point, theta: float = 0, tx: float = 0, ty: float = 0, shear: float = 0,
                                        zx: float = 1, zy: float = 1, fill_mode: str = 'nearest',
                                        c_val: float = 0., order: int = 1):
        theta = -theta
        tx = -tx
        ty = -ty
        shear = -shear
        zx = 1./zx
        zy = 1./zy

        final_affine_matrix, final_offset = _get_affine_transformation_matrix(self.size, theta, tx, ty, shear, zx, zy)
        point_arr = np.array(point)
        return tuple(np.matmul(final_affine_matrix, point_arr) + final_offset)

    def _get_crop_from_slice(self, key):
        top, bottom, left, right = 0, 0, 0, 0
        if type(key) == tuple:
            assert len(key) == 2, 'can only slice along columns, or columns and rows'
            if type(key[0]) == slice:
                assert (key[0].step is None) or (key[0].step == 1), 'step must be 1 or None when slicing'
                top = key[0].start if key[0].start is not None else 0
                top = self.size[1] + top if top < 0 else top
                bottom = key[0].stop if key[0].stop is not None else self.size[1]
                bottom = -bottom if bottom < 0 else self.size[1] - bottom
            elif key[0] == Ellipsis:
                pass
            if type(key[1]) == slice:
                assert (key[1].step is None) or (key[1].step == 1), 'step must be 1 or None when slicing'
                left = key[1].start if key[1].start is not None else 0
                left = self.size[0] + left if left < 0 else left
                right = key[1].stop if key[1].stop is not None else self.size[0]
                right = -right if right < 0 else self.size[0] - right
            elif key[0] == Ellipsis:
                pass
        elif key == Ellipsis:
            pass
        elif type(key) == slice:
            assert (key.step is None) or (key.step == 1), 'step must be 1 or None when slicing'
            top = key.start if key.start is not None else 0
            top = self.size[1] + top if top < 0 else top
            bottom = key.stop if key.stop is not None else 0
            bottom = self.size[1] + bottom if bottom < 0 else bottom
        return top, bottom, left, right

    def __getitem__(self, key):
        sliced_object = self.copy()
        top, bottom, left, right = self._get_crop_from_slice(key)
        if any((top, bottom, left, right)):
            sliced_object._crop(top, bottom, left, right)
        return sliced_object

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
            channel_mode=''.join(self.channel_mode),
            size=self.size,
            id=id(self))

    def _get_transform_config(self) -> list:
        return [{event['fn_name']: event['fn_args']} for event in self._transform_history]


def _transform_matrix_offset_center(matrix: np.ndarray, x: int, y: int) -> np.ndarray:
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def _get_affine_transformation_matrix(size: tuple, theta: float = 0, tx: float = 0, ty: float = 0, shear: float = 0,
                                      zx: float = 1, zy: float = 1) -> tuple:
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        w, h = size
        transform_matrix = _transform_matrix_offset_center(
            transform_matrix, h, w)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        return final_affine_matrix, final_offset
    else:
        return None, None


def _apply_affine_transform(x: np.ndarray, final_affine_matrix: np.ndarray, final_offset: np.ndarray,
                            fill_mode: str = 'nearest', c_val: float = 0., order: int = 1) -> np.ndarray:
    """Applies an affine transformation specified by the parameters given.
        Based on https://github.com/keras-team/keras-preprocessing/
            blob/master/keras_preprocessing/image/affine_transformations.py
    # Arguments
        x: 2D numpy array, single image.
        final_affine_matrix: afine transformation matrix

        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        c_val: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order: int, order of interpolation
    # Returns
        The transformed version of the input.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')

    if final_affine_matrix is not None and final_offset is not None:
        x = np.rollaxis(x, 2, 0)

        channel_images = [ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=c_val) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, 2 + 1)
    return x


def _repeat_c_channels(image):
    if image.ndim == 2:
        return np.repeat(np.expand_dims(image, axis=2), repeats=3, axis=2)
    elif image.ndim == 3 and image.shape[-1] == 1:
        return np.repeat(image, repeats=3, axis=2)
    elif image.ndim == 3 and image.shape[-1] == 3:
        return image
    else:
        raise ValueError('invalid image ndim {} and shape {}'.format(image.ndim, image.shape))
