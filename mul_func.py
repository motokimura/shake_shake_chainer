#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
from chainer import configuration

class Mul(chainer.function.Function):
    
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


def mul(x1, x2):
    return Mul()(x1, x2)
