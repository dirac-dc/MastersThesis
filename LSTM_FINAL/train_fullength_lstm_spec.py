import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.recurrent import LSTM
import keras.backend as K
K.set_image_dim_ordering('th')
import numpy as np
import sys
from os.path import isfile, join
import shutil
import h5py
import os.path
import glob
import time
from keras.layers import TimeDistributed
import stLSTM as ls

LIP_HEIGHT = 70
LIP_WIDTH = 100
LWM=LIP_WIDTH/2
LHM = LIP_HEIGHT/2
FRAME_ROWS = LIP_HEIGHT
FRAME_COLS = LIP_WIDTH
NFRAMES = 75
MARGIN = NFRAMES/2
COLORS = 1
CHANNELS = COLORS*NFRAMES
TRAIN_PER = 0.8
LR = 0.001
nb_pool = 2
BATCH_SIZE = 32
DROPOUT = 0.25
DROPOUT2 = 0.5
EPOCHS = 50
FINETUNE_EPOCHS = 10
activation_func2 = 'tanh'

respath = 'results/'
weight_path = join(respath,'weights/')
datapath = '/data/dccc/datasets/'

VIDFILE_NAME = 'viddata_nochann_lips.npy'
AUDFILE_NAME = 'auddata_spectogram_nochann.npy'
EXP ='FULL_LSTM_SPEC'

def load_data(datapath):
    viddata_path = join(datapath,VIDFILE_NAME)
    auddata_path = join(datapath,AUDFILE_NAME)
    if isfile(viddata_path) and isfile(auddata_path):
        print ('Loading data...')
        viddata = np.load(viddata_path)
        auddata = np.load(auddata_path)
        viddata = np.expand_dims(viddata,axis=2)
        vidctr = len(auddata)
        print ('Done.')
        print('vid shape','aud shape',viddata.shape,auddata.shape)
        return viddata, auddata
    else:
        print ('Preprocessed data not found.')
        sys.exit()

def split_data(viddata, auddata):
    vidctr = len(auddata)
    Xtr = viddata[:int(vidctr*TRAIN_PER),:,:,:]
    Ytr = auddata[:int(vidctr*TRAIN_PER),:]
    Xte = viddata[int(vidctr*TRAIN_PER):,:,:,:]
    Yte = auddata[int(vidctr*TRAIN_PER):,:]
    return (Xtr, Ytr), (Xte, Yte)

def savedata(Ytr, Ytr_pred, Yte, Yte_pred, respath=respath):
    np.save(join(respath,EXP+'_Ytr.npy'),Ytr)
    np.save(join(respath,EXP+'_Ytr_pred.npy'),Ytr_pred)
    np.save(join(respath,EXP+'_Yte.npy'),Yte)
    np.save(join(respath,EXP+'_Yte_pred.npy'),Yte_pred)

def standardize_data(Xtry, Ytr, Xtey, Yte):
    xtrain_mean = np.mean(Xtry).astype('float32')
    Xtrain=Xtry.astype('float32') #will only work on cluster
    Xtest=Xtey.astype('float32')
    Xtest = Xtest-xtrain_mean
    for i in range(len(Xtrain)):
        Xtrain[i,:,:,:] = (Xtrain[i,:,:,:]-xtrain_mean)/255.0
    Xtest = Xtest/255.0
    Y_means = np.mean(Ytr,axis=0)
    print('ytr', Ytr[0:10])
    Y_stds = np.std(Ytr, axis=0)
    for i in range(len(Y_stds)):
        if Y_stds[i] == 0:
            Y_stds[i] = 1
    # print('Standard Deviations',Y_stds)
    Ytr_norm = ((Ytr-Y_means)/Y_stds)
    Yte_norm = ((Yte-Y_means)/Y_stds)
    Ytr_norm = Ytr_norm.astype('float32')
    Yte_norm = Yte_norm.astype('float32')
    return Xtrain, Ytr_norm, Xtest, Yte_norm, Y_means, Y_stds

def train_net(model, Xtr, Ytr_norm, Xte, Yte_norm, batch_size=BATCH_SIZE, epochs=EPOCHS, finetune=False):
    if finetune:
        newest = max(glob.iglob(weight_path+'*.hdf5'), key=os.path.getctime)
        model.load_weights(newest)
        lr = LR/10
        foo='_FINETUNE_'
    else:
        lr = LR
        foo='_NORMAL_'
    adam = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=adam)
    checkpointer = ModelCheckpoint(filepath=weight_path+EXP+'weights.{epoch:02d}-{val_loss:.4f}.hdf5',
    monitor='val_loss', verbose=1, save_best_only=True)
    history = model.fit(Xtr, Ytr_norm, batch_size=16, nb_epoch=epochs,
    verbose=1, validation_data=(Xte, Yte_norm),callbacks=[checkpointer])
    newest = max(glob.iglob(weight_path+'*.hdf5'), key=os.path.getctime)
    model.load_weights(newest)
    try:
        np.save(join(respath,EXP+foo+'error_epoch.npy'),history.history)
    except:
        print('Could not save file: error_epoch.npy')
    return model

def predict(model, X, Y_means, Y_stds, batch_size=BATCH_SIZE):
    Y_pred = model.predict(X, batch_size=batch_size, verbose=1)
    Y_pred = (Y_pred*Y_stds+Y_means)
    return Y_pred

def main():
    start = time.time()
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    viddata, auddata = load_data(datapath)
    (Xtr,Ytr), (Xte, Yte) = split_data(viddata, auddata)
    net_out = Ytr.shape[1]
    Xtr, Ytr_norm, Xte, Yte_norm, Y_means, Y_stds = standardize_data(Xtr, Ytr, Xte, Yte)
    print('TYPES:',isinstance(Xtr[0,0,0,0,0],np.float64), isinstance(Ytr_norm[0,0],np.float64) )
    # stop = input('Exit ???')
    # if stop == 'y':
    #     sys.exit()
    model = ls.build_fulllstm_model(net_out = 148*640,FRAME_ROWS=70,FRAME_COLS=100)
    print(model.summary())
    model = train_net(model, Xtr, Ytr_norm, Xte, Yte_norm)
    model = train_net(model, Xtr, Ytr_norm, Xte, Yte_norm, epochs=FINETUNE_EPOCHS, finetune=True)
    Ytr_pred = predict(model, Xtr, Y_means, Y_stds)
    Yte_pred = predict(model, Xte, Y_means, Y_stds)
    savedata(Ytr, Ytr_pred, Yte, Yte_pred)
    print('finished saving')
    end = time.time()
    f = open('TIME_TAKEN_'+str(round((start-end)/3600.0,2))+'.txt','w')
    f.close()
if __name__ == "__main__":
    main()
