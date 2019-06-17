# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 21:56:28 2018

@author: Hasan
"""

import os
from model import Methods
from utils_VDSR import load_test
from utils_VDSR import psnr, ssim
import cv2
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
image_size = None
label_size = None
color_dim = 1
scale = 3

methodName ='VDSR'
test_folder = 'test'
loss = 'perceptual'
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
    
    weight_filename = '../srcnn/S3Models/VDSR/L1/model0049.hdf5'
    model = method.VDSR()
    model.load_weights(weight_filename)
    

              
    for img in X_test:
        img = img.astype('float')/255

        test_sample = img.reshape(1,img.shape[0],img.shape[1],color_dim)
        time_start = time.clock()
        predicted = model.predict(test_sample)
        time_elapsed = (time.clock() - time_start)
        print('Time:', time_elapsed)
        predicted = model.predict(img.reshape(1,img.shape[0],img.shape[1],color_dim))
        #predicted = np.clip(predicted, 0, 1)
        predicted_list.append(predicted.reshape(predicted.shape[1],predicted.shape[2],color_dim))
    
    n_img = len(predicted_list)
    dirname = 'result'
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
        cnn = np.float32((predicted_list[i]*255))
        if color_dim>1:
            cnn = cnn[:,:,0]
            bic = bic[:,:,0]
            gnd = gnd[:,:,0]
            
        bic = bic[scale: -scale, scale: -scale]
        gnd = gnd[scale: -scale, scale: -scale]
        cnn = cnn[scale: -scale, scale: -scale]
        cnn = np.clip(cnn, 0,255);
#        bic = bic[6: -6, 6: -6]
#        gnd = gnd[6: -6, 6: -6]
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
