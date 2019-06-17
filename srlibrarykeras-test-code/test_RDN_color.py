# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 21:56:28 2018

@author: Hasan
"""

import os
from model import Methods
from utils_RDN import load_test
from utils import psnr, ssim
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import scipy.io as sio

image_size = None
label_size = None

scale = 3

image_size_trained = 32
label_size_trained = image_size_trained*scale
padding = np.int(np.abs(image_size_trained-label_size_trained)/2)

loss_lists = ['cauchy', 'charbonnier', 'dssimloss', 'euclidean', 'fair', 'huber', 'L1', 'perceptual']
#loss_lists = ['perceptual']

methodName ='RDN'
test_folder = 'Test2'
#srcnn_cnn_accs_v1 = sio.loadmat('SRCNN3_Urban100_cnn_accs.mat')
#srcnn_cnn_accs_v2 = srcnn_cnn_accs_v1.get('SRCNN3_Urban100_cnn_accs')
srcnn_cnn_accs = np.zeros( (len(loss_lists)+1,2), np.float32)  

def main():
    ii = 1
    color_dim = 1
   
    for loss in loss_lists:
       print('loss:',loss)
       if loss=='perceptual':
           color_dim = 3 
       else:
           color_dim = 1 
        
       method = Methods(
            scale=scale,
            image_size=image_size,
            label_size=label_size,
            color_dim=color_dim,
            is_training=False,
        )
    
        
       X_pre_test, X_test, Y_test = load_test(scale=scale, test_folder=test_folder, color_dim=color_dim)
                 
       predicted_list = []
        
       weight_filename = '../srcnn/S'+str(scale)+'Models/'+methodName+'/'+loss+'/model0040.hdf5'
       model = method.RDN()
       model.load_weights(weight_filename)
        
                      
       for img in X_pre_test:
           img = img.astype('float')
           
           test_sample = img.reshape(1,img.shape[0],img.shape[1],color_dim)
           predicted = model.predict(test_sample)
           predicted_list.append(predicted.reshape(predicted.shape[1],predicted.shape[2],color_dim))
        
       n_img = len(predicted_list)
       dirname = 'resultNew/S3_'+methodName+'/'+loss
       if not os.path.exists(dirname):
            os.makedirs(dirname)
            
       psnr_bic = 0
       psnr_cnn = 0
       ssim_bic = 0
       ssim_cnn = 0
        
       dirname_gnd = './TestDatasets/'+test_folder
       img_list = os.listdir(dirname_gnd)
       for i in range(n_img):
           imgname = 'image{:02}'.format(i)
           low_res =  X_pre_test[i]
           bic = np.float32(X_test[i])
           gnd = np.float32(Y_test[i])
           cnn = np.float32((predicted_list[i]*1))
           
           if color_dim>1:
               cnn = cnn[:,:,0]
               gnd = gnd[:,:,0]
               bic = bic[:,:,0]
           
           cnn = cnn[scale:-scale, scale:-scale]
           bic = bic[scale:-scale, scale:-scale]
           gnd = gnd[scale:-scale, scale:-scale] 
            
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
     
       psnr_bic_m = psnr_bic/n_img
       psnr_cnn_m = psnr_cnn/n_img
       ssim_bic_m = ssim_bic/n_img
       ssim_cnn_m = ssim_cnn/n_img
       
       srcnn_cnn_accs[0][0] = psnr_bic_m
       srcnn_cnn_accs[0][1] = ssim_bic_m
       srcnn_cnn_accs[ii][0] = psnr_cnn_m
       srcnn_cnn_accs[ii][1] = ssim_cnn_m
       ii += 1
       sio.savemat(methodName+str(scale)+'_'+test_folder+'_cnn_accs', {methodName+str(scale)+'_'+test_folder+'_cnn_accs':srcnn_cnn_accs}, appendmat=True)  
        #####################
       print('psnr_bic_m:',psnr_bic_m)
       print('psnr_cnn_m:',psnr_cnn_m)
       print('ssim_bic_m:',ssim_bic_m)
       print('ssim_cnn_m:',ssim_cnn_m)

main()
