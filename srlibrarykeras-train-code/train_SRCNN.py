from model import Methods
from utils_SRCNN import load_train
import argparse
#if the pixels range is between 0-255 then use myLosses_0_255.py
#otherwise, if  the pixels range is between 0-1 then use myLosses.py
from myLosses import LossMethods as ls
from keras.callbacks import ModelCheckpoint, LearningRateScheduler 
from keras import optimizers
from keras.models import Sequential, model_from_json
from keras.utils import multi_gpu_model
import os
import math


##SRCNN
image_size = 57
label_size = 41

#for perceptual loss: color_dim=3
#for other loss : color_dim=1
color_dim=3
lr_rate=1e-4
decay_rate = 0.0

batch_size=128
epochs = 20

stride=10
scale=3
save_period = 1
#geman loss lr_rate = 0.1
#others loss lr_rate =  1e-4
loss_lists = ['L1','cauchy', 'charbonnier', 'euclidean', 'fair', 'huber',  'talwar', 'tukey','welsch', 'dssimloss']
#loss_lists = ['perceptual','phuber','logistic','dssimloss', 'geman']
#loss_lists = ['geman']
#loss_lists = ['L1']
loss_category = 'other' # 'other' or 'perceptual' 
#for loss in loss_lists:
#    print(loss)
   
methodName='SRCNN'


def main():
   
    print('-----Step1:loading dataset--------')
    X_train, Y_train = load_train(image_size=image_size, label_size=label_size, stride=stride, scale=scale, loss=loss_category, methodName=methodName)
    print('-----dataset loaded--------')
    print('-----Step2:normalizing data------')
    X_train = X_train.astype('float')/255
    Y_train = Y_train.astype('float')/255
    
    method = Methods(
        scale=scale,
        image_size=image_size,
        label_size=label_size,
        color_dim=color_dim,
        is_training=True,
        learning_rate=lr_rate,
        batch_size=batch_size,
        epochs=epochs)

    for loss in loss_lists:
        learning_rate = lr_rate
          
    
        print('-----Step3:training--------')
       
        #        #retraining a pretrained model
#        weight_filename = 'S3Models/SRCNN/L1/model0020.hdf5'
#        model = method.VDSR()
#        model.load_weights(weight_filename)
        
        model = method.SRCNN()
        model.summary()
   
        #return
        
        if loss=='cauchy':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.cauchy, metrics=['accuracy'])
        elif loss=='charbonnier':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.charbonnier, metrics=['accuracy'])  
        elif loss=='euclidean':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss="mean_squared_error", metrics=['accuracy'])
        elif loss=='fair':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.fair, metrics=['accuracy'])
        elif loss=='geman':
            print('---------',loss,'----------')
            learning_rate = 0.1
            opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            #opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.geman, metrics=['accuracy'])
        elif loss=='huber':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.huber, metrics=['accuracy'])  
        elif loss=='L1':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss="mean_absolute_error", metrics=['accuracy'])
        elif loss=='talwar':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.talwar, metrics=['accuracy'])
        elif loss=='tukey':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.tukey, metrics=['accuracy'])
        elif loss=='welsch':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.welsch, metrics=['accuracy'])
        elif loss=='perceptual':
            print('---------',loss,'----------')
            ls.initparams()
         
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.perceptual, metrics=['accuracy'])
        elif loss=='dssimloss':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.dssimloss, metrics=['accuracy'])
        
        elif loss=='logistic':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.logistic, metrics=['accuracy'])
        
        elif loss=='phuber':
            print('---------',loss,'----------')
            #opt = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate, nesterov=False)
            opt = optimizers.Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=ls.phuber, metrics=['accuracy'])
         
        directory='S'+str(scale)+'Models/'+methodName+'/'+loss
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        callbacks_list = []
                    
        model_checkpoint = ModelCheckpoint(filepath=directory+'/model{epoch:04d}.hdf5',period=save_period)
        #lrate_checkpoint = LearningRateScheduler(step_decay)
        callbacks_list.append(model_checkpoint)
        #callbacks_list.append(lrate_checkpoint)
        history = model.fit(X_train, Y_train, shuffle = True, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1,  callbacks=callbacks_list)
    

#if __name__ == '__main__':
#    main(args=parser.parse_args())

main()