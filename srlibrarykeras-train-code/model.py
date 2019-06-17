import os
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras import losses as ls
from keras.utils import multi_gpu_model

from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(3)

from keras.layers import Conv2D,Convolution2D, Conv2DTranspose, LeakyReLU, Add, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.layers.core import Activation
from keras.layers import MaxPooling2D, Concatenate, Input, Lambda, merge, ZeroPadding2D, add, Add

from keras import regularizers
import tensorflow as tf
from subpixel import Subpixel, icnr_weights
from common import *

class Methods:
    def __init__(self, scale, image_size, label_size, color_dim, is_training,  learning_rate=1e-4, batch_size=128, epochs=1500):
        
       
        os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5,6,7' 
        self.scale = scale
        self.image_size = image_size
        self.label_size = label_size
        self.color_dim = color_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
    
    
    def SRCNN(self):
        model = Sequential()
        model.add(Conv2D(128,(9,9),padding='valid',input_shape=(self.image_size,self.image_size,self.color_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(64,(5,5),padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.color_dim,(5,5),padding='valid'))
        model = multi_gpu_model(model, gpus=7)
        return model
    
    def VDSR(self):
        
        print(self.color_dim)
        #reference: https://github.com/GeorgeSeif/VDSR-Keras/blob/master/vdsr.py
        input_img = Input(shape=(self.image_size,self.image_size,self.color_dim))

        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img) #01 
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #02
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #03
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #04
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #05
        model = Activation('relu')(model)
        
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #06
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #07
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #08
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #09
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #10
        model = Activation('relu')(model)
        
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #11
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #12
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #13
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #14
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #15
        model = Activation('relu')(model)
        
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #16
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #17
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #18
        model = Activation('relu')(model)
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model) #19
        model = Activation('relu')(model)
        model = Conv2D(self.color_dim, (3, 3), padding='same', kernel_initializer='he_normal')(model) #20
        res_img = model
        
        output_img = add([res_img, input_img])
        
        model = Model(input_img, output_img)
        model = multi_gpu_model(model, gpus=7)
        return model
    
    def EDSR(self):
        scale=self.scale
        color_dim = self.color_dim
        num_filters=64
        num_res_blocks=16
        res_block_scaling=None 
        tanh_activation=False
        x_in = Input(shape=(None, None, color_dim))
        x = Normalization()(x_in)
    
        x = b = Conv2D(num_filters, 3, padding='same')(x)
        for i in range(num_res_blocks):
            b = self.res_block(b, num_filters, res_block_scaling)
        
        b = Conv2D(num_filters, 3, padding='same')(b)
        x = Add()([x, b])
    
        x = self.upsample(x, scale, num_filters)
        x = Conv2D(self.color_dim, (3,3), padding='same')(x)
       
        if tanh_activation:
            x = Activation('tanh')(x)
            x = Denormalization_m11()(x)
        else:
            x = Denormalization()(x)
    
        model = Model(x_in, x, name="edsr")
        model = multi_gpu_model(model, gpus=7)
        #model.summary()
        return model
    
    
    def res_block(self, x_in, filters, scaling):
        x = Conv2D(filters, 3, padding='same')(x_in)
        x = Activation('relu')(x)
        x = Conv2D(filters, 3, padding='same')(x)
        x = Add()([x_in, x])
        
        if scaling:
            x = Lambda(lambda t: t * scaling)(x)
        return x
    
    
    def upsample(self, x, scale, num_filters):
        def upsample_1(x, factor, **kwargs):
            x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
            return SubpixelConv2D(factor)(x)
    
        if scale == 2:
            x = upsample_1(x, 2, name='conv2d_1_scale_2')
        elif scale == 3:
            x = upsample_1(x, 3, name='conv2d_1_scale_3')
        elif scale == 4:
            x = upsample_1(x, 2, name='conv2d_1_scale_2')
            x = upsample_1(x, 2, name='conv2d_2_scale_2')
    
        return x

    def upsample_RDN(self, x, scale, num_filters):
        def upsample_1_RDN(x, factor, **kwargs):
            x = Conv2D(num_filters * (factor ** 2), 3, padding='same', activation='relu',**kwargs)(x)
            return SubpixelConv2D(factor)(x)
    
        if scale == 2:
            x = upsample_1_RDN(x, 2, name='conv2d_1_scale_2')
        elif scale == 3:
            x = upsample_1_RDN(x, 3, name='conv2d_1_scale_3')
        elif scale == 4:
            x = upsample_1_RDN(x, 2, name='conv2d_1_scale_2')
            x = upsample_1_RDN(x, 2, name='conv2d_2_scale_2')
    
        return x    

    def RDN(self):
        channel = self.color_dim
        RDB_count=20
        scale = self.scale
        self.channel_axis = 3 
        inp = Input(shape = (None, None , channel))

        pass1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(inp)

        pass2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(pass1)

        
        RDB = self.RDBlocks(pass2 , 'RDB1')
        RDBlocks_list = [RDB,]
        for i in range(2,RDB_count+1):
            RDB = self.RDBlocks(RDB ,'RDB'+str(i))
            RDBlocks_list.append(RDB)
            
        out = Concatenate(axis = self.channel_axis)(RDBlocks_list)
        out = Conv2D(filters=64 , kernel_size=(1,1) , strides=(1,1) , padding='same')(out)
        out = Conv2D(filters=64 , kernel_size=(3,3) , strides=(1,1) , padding='same')(out)

        output = Add()([out , pass1])
        
        output = self.upsample_RDN(output, scale, 64)
        
#        if scale == 2:
#            output = Subpixel(64, (3,3), r = 2, padding='same',activation='relu')(output)
#        if scale == 3:
#            output = Subpixel(64, (3,3), r = 2, padding='same',activation='relu')(output)
#        if scale == 4:
#            output = Subpixel(64, (3,3), r = 2, padding='same',activation='relu')(output)
        
        output = Conv2D(filters =self.color_dim , kernel_size=(3,3) , strides=(1 , 1) , padding='same')(output)

        model = Model(inputs=inp , outputs = output)
        #model.summary()
        model = multi_gpu_model(model, gpus=7)
        return model

    def RDBlocks(self,x,name , count = 6 , g=32):
        ## 6 layers of RDB block
        ## this thing need to be in a damn loop for more customisability
        li = [x]
        pas = Conv2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu' , name = name+'_conv1')(x)
        
        for i in range(2 , count+1):
            li.append(pas)
            out =  Concatenate(axis = self.channel_axis)(li) # conctenated out put
            pas = Conv2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv'+str(i))(out)
        
        # feature extractor from the dense net
        li.append(pas)
        out = Concatenate(axis = self.channel_axis)(li)
        feat = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out)
        
        feat = Add()([feat , x])
        return feat
        
