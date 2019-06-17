# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 21:56:28 2018

@author: Hasan
"""

import os
from model import Methods
from utils_EDSR import load_test
from utils_EDSR import psnr, ssim
import cv2
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy import misc


image_size = None
label_size = None
color_dim = 1
scale = 3

image_size_trained = 48
label_size_trained = image_size_trained*scale
padding = np.int(np.abs(image_size_trained-label_size_trained)/2)

test_folder = 'Set5'
def main():
  

    method = Methods(
        scale=scale,
        image_size=image_size,
        label_size=label_size,
        color_dim=color_dim,
        is_training=False,
       )

    
    X_pre_test, X_test, Y_test = load_test(scale=scale, test_folder=test_folder, color_dim=color_dim)
 
    predicted_list = []
    
    weight_filename = '../srcnn/S'+str(scale)+'Models/EDSR/logistic/model0001.hdf5'
    model = method.EDSR()
    model.load_weights(weight_filename)
    

              
    for img in X_pre_test:
        img = img.astype('float')
        h1 = img.shape[0]
        w1 = img.shape[1] 
        h0 = np.int(h1/scale)
        w0 = np.int(w1/scale) 

        img = img[:,:,0]
        #img = img/255 
        predicted = model.predict(img.reshape(1,img.shape[0],img.shape[1],1))
        
        predicted_list.append(predicted.reshape(predicted.shape[1],predicted.shape[2],1))
    
    n_img = len(predicted_list)
    dirname = 'result3'
    psnr_bic = 0
    psnr_cnn = 0
    ssim_bic = 0
    ssim_cnn = 0
    
    dirname2 = './TestDatasets/'+test_folder
    img_list = os.listdir(dirname2)
    for i in range(n_img):
        imgname = 'image{:02}'.format(i)
        low_res =  X_pre_test[i]
        bic = np.float32(X_test[i])
        gnd = np.float32(Y_test[i])
        cnn = predicted_list[i]
        
        cnn = np.float32((cnn*1))
    
        cnn = cnn[scale:-scale, scale:-scale, :]
        bic = bic[scale:-scale, scale:-scale, :]
        gnd = gnd[scale:-scale, scale:-scale, :]
#    
        name = os.path.splitext(os.path.basename(img_list[i]))[0]
       
        #cv2.imwrite(os.path.join(dirname,imgname+'_original.bmp'), low_res)
        cv2.imwrite(os.path.join(dirname,name+'_bic.bmp'), bic)
        cv2.imwrite(os.path.join(dirname, name+'_gnd.bmp'), gnd)
        cv2.imwrite(os.path.join(dirname,name+'_cnn.bmp'), cnn)
 
        bic_ps =  psnr(gnd, bic)
        cnn_ps = psnr(gnd, cnn)
        bic_ss  = ssim(gnd, bic)
        cnn_ss  = ssim(gnd, cnn)
        print('bic:',bic_ps)
        print('cnn',cnn_ps)       
        
        psnr_bic += bic_ps
        psnr_cnn += cnn_ps
        ssim_bic += bic_ss
        ssim_cnn += cnn_ss
#     
    #####################
    print('psnr_bic:',psnr_bic/n_img)
    print('psnr_cnn:',psnr_cnn/n_img)
    print('ssim_bic:',ssim_bic/n_img)
    print('ssim_cnn:',ssim_cnn/n_img)

main()
