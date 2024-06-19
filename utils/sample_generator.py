import os
import cv2
import random

import numpy as np

from color_aug import color_augmentation
from geometric_aug import rotation_augumentation


def samples(image, keypoint2d, method):
    if method == 0:
        return image, keypoint2d
    elif method == 1:
        return color_augmentation(image), keypoint2d
    else:
        return rotation_augumentation(image, keypoint2d)
