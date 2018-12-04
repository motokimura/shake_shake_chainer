#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
from chainer import configuration
from chainer import Variable
import chainer.functions as F
import chainer.links as L

import cupy
import numpy as np


class ShakeShake(chainer.function.Function):
    
    def __init__(self):
        return

    def forward(self, inputs):
        x1, x2 = inputs
        xp = cuda.get_array_module(x1) # Get numpy(x=n) or cupy(x=c) array module
        alpha = xp.ones(x1.shape, dtype=x1.dtype) * 0.5
        
        if configuration.config.train:
            for i in range(len(alpha)):
                alpha[i] = xp.random.rand()

        return x1 * alpha + x2 * (xp.ones(x1.shape, dtype=x1.dtype) - alpha),
    
    def backward(self, inputs, grad_outputs):
        gx, = grad_outputs
        xp = cuda.get_array_module(gx)
        beta = xp.empty(gx.shape, dtype=gx.dtype)

        for i in range(len(beta)):
            beta[i] = xp.random.rand()
        
        return gx * beta, gx * (xp.ones(gx.shape, dtype=gx.dtype) - beta)

def shake_shake(x1, x2):
    return ShakeShake()(x1, x2)


class ShortCut(chainer.link.Chain):
    
    def __init__(self, in_channels, out_channels, stride, initialW):
        super(ShortCut, self).__init__()
        
        with self.init_scope():
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
        h0 = F.relu(x)

        h1 = h0
        h1 = F.average_pooling_2d(h1, ksize=1, stride=self._stride, pad=0)
        h1 = self.conv1(h1)

        h2 = self._zero_pads(self._zero_pads(h0, pad=1, axis=2), pad=1, axis=3)
        h2 = F.average_pooling_2d(h2, ksize=1, stride=self._stride, pad=0)[:, :, 1:, 1:]
        h2 = self.conv2(h2)

        h = F.concat((h1, h2), axis=1)
        h = self.bn(h)
        return h
        
    def _zero_pads(self, x, pad, axis):
        sizes = list(x.data.shape)
        sizes[axis] = pad

        xp = cuda.get_array_module(x) # Get numpy(x=n) or cupy(x=c) array module
        dtype = x.dtype
        pad_tensor = xp.zeros(sizes, dtype=dtype)
        if isinstance(x.data, cupy.core.core.ndarray):
            device = cuda.get_device_from_array(x.data)
            pad_tensor = cuda.to_gpu(pad_tensor, device=device)

        pad_tensor = Variable(pad_tensor)
        return F.concat((pad_tensor, x), axis=axis)