import cv2
from PIL import Image
import os
import numpy as np


IMAGE_BE = os.environ.get('NYAN_IMAGE_BE', 'PIL')

if IMAGE_BE == 'PIL':
    def IMREAD_FN(x):
        return np.array(Image.open(x).convert('RGB')).astype(np.uint8)[:, :, ::-1]
elif IMAGE_BE == 'cv2':
    def IMREAD_FN(x):
        return cv2.imread(x).astype(np.uint8)
else:
    raise NotImplementedError('IMAGE_BE {} not implemented'.format(IMAGE_BE))
