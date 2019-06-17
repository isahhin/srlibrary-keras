#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:26:34 2019

@author: root
"""

import numpy as np
import tensorflow as tf

from keras.layers import Lambda

#DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

DIV2K_RGB_MEAN = np.array([0.4381]) * 255



def SubpixelConv2D(scale, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)

def Normalization(rgb_mean=DIV2K_RGB_MEAN, **kwargs):
    return Lambda(lambda x: (x - rgb_mean) / 127.5, **kwargs)

def Denormalization(rgb_mean=DIV2K_RGB_MEAN, **kwargs):
    return Lambda(lambda x: x * 127.5 + rgb_mean, **kwargs)

def Normalization_01(**kwargs):
    return Lambda(lambda x: x / 255, **kwargs)

def Normalization_m11(**kwargs):
    return Lambda(lambda x: x / 127.5 - 1, **kwargs)

def Denormalization_m11(**kwargs):
    return Lambda(lambda x: (x + 1) * 127.5, **kwargs)