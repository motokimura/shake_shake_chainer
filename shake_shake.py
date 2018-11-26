#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L
from chainer.initializers import normal
import chainer.links.model.vision.resnet as R
import collections

from mul_func import mul


class BuildingShakeBlocks(chainer.link.Chain):

    """Build a stage that consists of several residual/shake blocks.
    Args:
        n_block (int): Number of residual/shake blocks used in the stage.
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
    """

    def __init__(self, n_block, in_channels, out_channels, stride, initialW=None):
        super(BuildingShakeBlocks, self).__init__()
        with self.init_scope():
            downsample = ShakeBlockA(in_channels, out_channels, stride, initialW)
            self.a = downsample
            self._blocks = ['a']
            for i in range(n_block - 1):
                name = 'b{}'.format(i + 1)
                basic = ShakeBlockB(out_channels, initialW)
                setattr(self, name, basic)
                self._blocks.append(name)
    
    def __call__(self, x):
        for name in self._blocks:
            block = getattr(self, name)
            x = block(x)
        return x


class ShakeBlockA(chainer.link.Chain):

    def __init__(self, in_channels, out_channels, stride=2, initialW=None):
        super(ShakeBlockA, self).__init__()

        with self.init_scope():
            self.branch1 = RCBRCB(in_channels, out_channels, stride=stride, initialW=initialW)
            self.branch2 = RCBRCB(in_channels, out_channels, stride=stride, initialW=initialW)

            self.conv1 = L.Convolution2D(
                in_channels, out_channels // 2, 
                ksize=1, stride=1, pad=0, initialW=initialW, nobias=True
            )
            self.conv2 = L.Convolution2D(
                in_channels, out_channels // 2, 
                ksize=1, stride=1, pad=0, initialW=initialW, nobias=True
            )
            self.bn = L.BatchNormalization(out_channels)
        
        self._stride = stride
            
    def __call__(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)

        x0 = F.relu(x)
        x1 = x0
        x1 = self.conv1(F.average_pooling_2d(x1, ksize=1, stride=self._stride, pad=0))
        x2 = self._zero_pads(self._zero_pads(x0, pad=1, axis=2), pad=1, axis=3)
        x2 = self.conv2(F.average_pooling_2d(x2, ksize=1, stride=self._stride, pad=0)[:, :, 1:, 1:])
        h0 = F.concat((x1, x2), axis=1)
        h0 = self.bn(h0)

        return mul(h1, h2) + h0
    
    def _zero_pads(self, x, pad, axis):
        sizes = list(x.data.shape)
        sizes[axis] = pad

        xp = cuda.get_array_module(x)
        dtype = x.dtype
        pad_tensor = xp.zeros(sizes, dtype=dtype)
        if type(x.data) != np.ndarray:
            device = cuda.get_device_from_array(x.data)
            pad_tensor = cuda.to_gpu(pad_tensor, device=device)

        pad_tensor = Variable(pad_tensor)
        return F.concat((pad_tensor, x), axis=axis)


class ShakeBlockB(chainer.link.Chain):

    def __init__(self, in_channels, initialW=None):
        super(ShakeBlockB, self).__init__()
        with self.init_scope():
            self.branch1 = RCBRCB(in_channels, in_channels, initialW=initialW)
            self.branch2 = RCBRCB(in_channels, in_channels, initialW=initialW)

    def __call__(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        return  mul(h1, h2) + x


class RCBRCB(chainer.link.Chain):

    def __init__(self, in_channels, out_channels, stride=1, initialW=None):
        super(RCBRCB, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels, out_channels, 
                ksize=3, stride=stride, pad=1, initialW=initialW, nobias=True
            )
            self.bn1 = L.BatchNormalization(out_channels)
            self.conv2 = L.Convolution2D(
                out_channels, out_channels,
                ksize=3, stride=1, pad=1, initialW=initialW, nobias=True
            )
            self.bn2 = L.BatchNormalization(out_channels)
    
    def __call__(self, x):
        h = self.bn1(self.conv1(F.relu(x)))
        h = self.bn2(self.conv2(F.relu(h)))
        return h


class ShakeShake(chainer.Chain):

    def __init__(self, n_out=10, n_layer=26, base_width=32):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        if (n_layer - 2) % 6 == 0:
            n_blocks = [(n_layer - 2) // 6] * 3
        else:
            raise ValueError(
                'The n_layer argument should be mod({} - 2, 6) == 0, \
                 but {} was given.'.format(n_layer, n_layer))
        
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 16,
                ksize=3, stride=1, pad=1, nobias=True, **kwargs
            )
            self.bn1 = L.BatchNormalization(16)

            k = base_width
            self.res2 = BuildingShakeBlocks(
                n_blocks[0], 16, k, 1, **kwargs)
            self.res3 = BuildingShakeBlocks(
                n_blocks[1], k, 2 * k, 2, **kwargs)
            self.res4 = BuildingShakeBlocks(
                n_blocks[2], 2 * k, 4 * k, 2, **kwargs)
            
            self.fc5 = L.Linear(4 * k, n_out)

    def __call__(self, x):
        h = x # [b, 3, 32, 32]
        h = self.bn1(self.conv1(h)) # [b, 16, 32, 32]
        h = self.res2(h) # [b, 32, 32, 32]
        h = self.res3(h) # [b, 64, 16, 16]
        h = self.res4(h) # [b, 128, 8, 8]
        B, _, H, W = h.data.shape
        h = F.average_pooling_2d(F.relu(h), ksize=(H, W)).reshape(B, -1) # [b, 128]
        h = self.fc5(h) # [b, n_out]
        return h