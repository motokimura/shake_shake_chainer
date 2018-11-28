#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer.training import extension
import math

class LrSceduler_CosineAnneal(extension.Extension):

    trigger = (1, 'epoch')

    def __init__(self, base_lr, epochs, optimizer_name='main', lr_name='lr'):
        
        self._base_lr = base_lr
        self._epochs = epochs
        self._optimizer_name = optimizer_name
        self._lr_name = lr_name

    def __call__(self, trainer):

        e = trainer.observation['epoch']
        lr = 0.5 * self._base_lr * (math.cos(math.pi * e / self._epochs) + 1.)

        optimizer = trainer.updater.get_optimizer(self._optimizer_name)
        setattr(optimizer, self._lr_name, lr)