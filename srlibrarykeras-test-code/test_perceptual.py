# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 21:56:28 2018

@author: Hasan
"""

import os
from model import Methods
from utils_perceptual import load_test
from utils_perceptual import psnr, ssim
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
image_size = None
label_size = None
color_dim = 3
scale = 3


def main():
    method = Methods(
        scale=scale,
        label_size=label_size,
        image_size=image_size,
        color_dim=color_dim,
        is_training=False)
    
    X_pre_test, X_test, Y_test = load_test(scale=scale)
      
    
    predicted_list = []
    
    weight_filename = '../srcnn/S3Models/SRCNN/perceptual/model0020.hdf5'
    model = method.SRCNN()
    model.load_weights(weight_filename)
    

              
    for img in X_test:
        img = img.astype('float')/255
        
        predicted = model.predict(img.reshape(1,img.shape[0],img.shape[1],3))
        predicted_list.append(predicted.reshape(predicted.shape[1],predicted.shape[2],3))
    
    n_img = len(predicted_list)
    dirname = 'result'
    psnr_bic = 0
    psnr_cnn = 0
    ssim_bic = 0
    ssim_cnn = 0
    
    dirname2 = './test'
    img_list = os.listdir(dirname2)
    for i in range(n_img):
        imgname = 'image{:02}'.format(i)
        low_res =  X_pre_test[i]
        bic = np.float32(X_test[i])
        gnd = np.float32(Y_test[i])
        cnn = np.float32((predicted_list[i]*255))
        cnn = cnn[:,:,0]
        gnd = gnd[:,:,0]
        bic = bic[:,:,0]
   
        bic = bic[8: -8, 8: -8]
        gnd = gnd[8: -8, 8: -8]
        
#        bic = bic[6: -6, 6: -6]
#        gnd = gnd[6: -6, 6: -6]
        name = os.path.splitext(os.path.basename(img_list[i]))[0]
       
        #cv2.imwrite(os.path.join(dirname,imgname+'_original.bmp'), low_res)
        cv2.imwrite(os.path.join(dirname,name+'_bic.bmp'), bic)
        cv2.imwrite(os.path.join(dirname, name+'_gnd.bmp'), gnd)
        cv2.imwrite(os.path.join(dirname,name+'_cnn.bmp'), cnn)
#        bic_ps =  cv2.PSNR(gnd, bic)
#        cnn_ps = cv2.PSNR(gnd, cnn)
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
     
    #####################
    print('psnr_bic:',psnr_bic/n_img)
    print('psnr_cnn:',psnr_cnn/n_img)
    print('ssim_bic:',ssim_bic/n_img)
    print('ssim_cnn:',ssim_cnn/n_img)

main()
