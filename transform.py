#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainercv import transforms
import numpy as np


def transform(inputs, mean, std, crop_size=(32, 32), pad=4, train=True):
    img, label = inputs
    img = img.copy()

    if train:
        # Random crop
        img = np.pad(img, pad_width=pad, mode='constant')
        img = transforms.random_crop(img, tuple(crop_size))

        # Random flip
        img = transforms.random_flip(img, x_random=True)
    
    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]
    
    return img, label
