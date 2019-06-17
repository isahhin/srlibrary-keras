import os
import cv2
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot  as plt
from scipy import misc
import math 
from matlab_imresize import imresize as mat_imresize
#reference: https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float64)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
        rlt = rlt.astype(in_img_type)
    return rlt


def load_train(image_size=33, label_size=21, stride=14, scale=3, loss=None, methodName=None):
    
    
    #padding = np.int(np.abs((image_size-label_size)/2))
    padding = 0
    print('padding:',padding)
    dirname = './DIV2K'
    dir_list = os.listdir(dirname)
   
    labels = []
    for file in dir_list:
        img = cv2.imread(os.path.join(dirname,file), cv2.IMREAD_COLOR)
        img = rgb2ycbcr(img)
        img = img[0:img.shape[0]-np.remainder(img.shape[0],scale),0:img.shape[1]-np.remainder(img.shape[1],scale)]
        labels.append(img)
  
   
   
    trains = []
    for img in labels:
        h1 = img.shape[0]
        w1 = img.shape[1]
        h0 = np.int(h1/scale)
        w0 = np.int(w1/scale)   
        img = np.float32(img)
        pil_image = mat_imresize(img, output_shape=(h0,w0))        
        img = np.array(pil_image)
       
        trains.append(img)
      
    print('extracting sub patches')
    sub_trains = []
    sub_labels = []

    if loss=='perceptual':
        
        nn = 151300
        sub_labels = np.zeros((nn,label_size, label_size, 3), dtype='float32')
        i = 0
        for train, label in zip(trains, labels):
            v, h = label.shape   
            for x in range(0,v-label_size+1,stride):              
                for y in range(0,h-label_size+1,stride):
                    #print([x,y])                  
                    sub_label = label[x:x+label_size,y:y+label_size]
                    sub_label = sub_label.reshape(label_size,label_size,1)
                    sub_label = np.concatenate([sub_label, sub_label, sub_label],2)
                    #sub_labels.append(sub_label)
                    sub_labels[i,:,:,:] = sub_label
                    i += 1
                    
        
        print('iiiii:',i)
        
        labels = []
        ii=0
        for file in dir_list:
            labels.append(ii)
        
    
        sub_trains = np.zeros((nn,image_size, image_size, 3), dtype='float32')
        i = 0
        for train, label in zip(trains, labels):
            v, h = train.shape   
            for x in range(0,v-image_size+1,np.int(stride/scale)):              
                for y in range(0,h-image_size+1,np.int(stride/scale)):
                    #print([x,y])                  
                    sub_train = train[x:x+image_size,y:y+image_size]
                    sub_train = sub_train.reshape(image_size,image_size,1)
                    sub_train = np.concatenate([sub_train, sub_train, sub_train],2)
                    #sub_trains.append(sub_train)
                    sub_trains[i,:,:,:] = sub_train
                    i += 1
    
    else:
        for train, label in zip(trains, labels):
            v, h = label.shape   
            for x in range(0,v-label_size+1,stride):              
                for y in range(0,h-label_size+1,stride):
                    #print([x,y])                  
                    sub_label = label[x:x+label_size,y:y+label_size]
                    sub_label = sub_label.reshape(label_size,label_size,1)
                    sub_labels.append(sub_label)
        
        for train, label in zip(trains, labels):
            v, h = train.shape   
            for x in range(0,v-image_size+1,np.int(stride/scale)):              
                for y in range(0,h-image_size+1,np.int(stride/scale)):
                    #print([x,y])                  
                    sub_train = train[x:x+image_size,y:y+image_size]
                    sub_train = sub_train.reshape(image_size,image_size,1)
                    sub_trains.append(sub_train)
                
    del trains
    del labels
    print('size sub_labels:',len(sub_labels))
    sub_trains = np.array(sub_trains)   
    sub_labels = np.array(sub_labels)
    #sub_trains, sub_labels = shuffle(sub_trains, sub_labels, random_state=0)
    return sub_trains, sub_labels

def load_test(scale=3):
    dirname = './test'
    dir_list = os.listdir(dirname)
    images = [ rgb2ycbcr(cv2.imread(os.path.join(dirname,img), cv2.IMREAD_COLOR)) for img in dir_list]
    images = [img[0:img.shape[0]-np.remainder(img.shape[0],scale),0:img.shape[1]-np.remainder(img.shape[1],scale)] for img in images]

    tests = images.copy()
    #tests2 = images.copy()
    labels = images.copy()
    
    #pre_tests = images.copy()
    pre_tests = [ img for img in images]
    i = 0
    for img in tests:
        h1 = img.shape[0]
        w1 = img.shape[1]
        h0 = np.int(h1/scale)
        w0 = np.int(w1/scale)
        img = np.float32(img)
        pil_image = mat_imresize(img, output_shape=(h0,w0))
        img = np.array(pil_image)
       
        pre_tests[i]=img
        i += 1
    
    tests = [cv2.resize(img, None, fx=scale/1, fy=scale/1, interpolation=cv2.INTER_CUBIC) for img in pre_tests]

    i = 0
    for img in pre_tests:
        img2 = tests[i]
        h1 = img2.shape[0]
        w1 = img2.shape[1]
        img = np.float32(img)
        pil_image = mat_imresize(img, output_shape=(h1,w1))
        img = np.array(pil_image)
        
#        ret = np.empty((h1, w1, 3), dtype=np.uint8)
#        ret[:, :, 0] = img
#        ret[:, :, 1] = img
#        ret[:, :, 2] = img
#        img = ret.copy()
       
        tests[i]=img
        i += 1
    
    pre_tests = [img.reshape(img.shape[0],img.shape[1],1) for img in pre_tests]
    tests = [img.reshape(img.shape[0],img.shape[1],1) for img in tests]
    labels = [img.reshape(img.shape[0],img.shape[1],1) for img in labels]

    return pre_tests, tests, labels

def mse(y, t):
    return np.mean(np.square(y - t))

def psnr(target, ref):
    # assume RGB image
    target_data = np.array(target, dtype='float32')
    ref_data = np.array(ref, dtype='float32')

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)

def ssim(x, y):
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov = np.mean((x - mu_x) * (y - mu_y))
    c1 = np.square(0.01 * 255)
    c2 = np.square(0.03 * 255)
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))



