# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:01:19 2018

@author: Hasan
"""
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from keras import backend as K

from keras.models import Sequential, model_from_json

vgg_content = Sequential()
class LossMethods:
    def initparams():        
        inputsize = (None,None,3)
        vgg_content = VGG16(weights='imagenet', include_top=False, input_shape=inputsize)
        for l in vgg_content.layers: l.trainable=False
        
        vgg_content.summary()
    
    def cauchy(y_true, y_pred):
    
        C=1.0
        res = y_true - y_pred    
        yt = (0.5*(C**2) ) * tf.keras.backend.log(1+ ((res/C)**2) )
    
        return tf.keras.backend.mean(yt)
    
    
    def charbonnier(y_true, y_pred):
        epsilon = 1e-3
        res = y_true - y_pred
        yt = tf.keras.backend.sqrt((res**2) + (epsilon**2))
        
        return tf.keras.backend.mean(yt)
    
    #Euclidean
    #mean_squared_error
    
    def fair(y_true, y_pred):
        C=1.0
        res = y_true - y_pred
        yt = (C**2) * ( tf.keras.backend.abs(res)/C -  tf.keras.backend.log(1+  tf.keras.backend.abs(res)/C))
        
        return tf.keras.backend.mean(yt)
    
    def geman(y_true, y_pred):
        C=1.0
        res = y_true - y_pred
        yt = ((res**2) /2)/((C**2) +  (res**2))
       
        return tf.keras.backend.mean(yt)
    
    def huber(y_true, y_pred):
        C=1.0
        res = y_true - y_pred
        cond  = tf.keras.backend.abs(res) < C
    
        squared_loss = 0.5 * (res**2)
        linear_loss  = C * (tf.keras.backend.abs(res) - 0.5 * C)
    
        return tf.keras.backend.mean(tf.where(cond, squared_loss, linear_loss))
    
    #L1
    #mean_absolute_error
     
    def talwar(y_true, y_pred):
        #correct it
        res = y_true - y_pred 
        absRes = tf.keras.backend.abs(res) 
        C=1.0
        
        linearRegion = tf.greater_equal(absRes, C)
        yt =  tf.where (linearRegion, (C**2)*tf.ones_like(absRes), absRes**2) 
            
        return tf.keras.backend.mean(yt)
    
    def tukey(y_true, y_pred):
        #correct it
        
        res = y_true - y_pred 
        
        C=1.0
        scale = (C**2) / 6 ;
        yt = scale * (1 - (1 - (res / C)**2)**3) ;
        
        linearRegion = tf.greater(tf.abs(res), C)
        yt =  tf.where(linearRegion, scale*tf.ones_like(yt), yt) 
    
        return tf.keras.backend.mean(yt)
    
    def welsch(y_true, y_pred):
        #correct it
        res = y_true - y_pred 
        C=1.0
        yt = 0.5 * (C**2) * (1-tf.exp(-(res/C)**2)) 
            
        return tf.keras.backend.mean(yt)
    
    def logistic(y_true, y_pred):
        #correct it
        res = y_true - y_pred 
        C=1.0
        yt = (C**2) * tf.log(tf.cosh(res/C))
 
        return tf.keras.backend.mean(yt) 
    
    def phuber(y_true, y_pred):
        #correct it
        res = y_true - y_pred 
        C=1.0
        yt = (C**2)*(tf.sqrt(1+(res**2)/(C**2))-1);
        
        return tf.keras.backend.mean(yt) 
    
    def dssimloss(y_true, y_pred):
        ssim2 = tf.image.ssim(y_true, y_pred, 1.0)
        return K.mean(1-ssim2)
    
    def perceptual(y_true, y_pred):
    
        y_t=vgg_content(y_true)
        y_p=vgg_content(y_pred)
        
        per_loss = tf.losses.mean_squared_error(y_t,y_p)
        return tf.keras.backend.mean(per_loss)


