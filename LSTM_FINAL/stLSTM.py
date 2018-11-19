from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
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
from extra import TimeDistributedConvolution2D
from extra import TimeDistributedMaxPooling2D
# import LSTM_train as ls
# import procimages as pi


FRAME_ROWS = 70
FRAME_COLS = 100
NFRAMES = 75
MARGIN = NFRAMES/2
COLORS = 1
CHANNELS = COLORS*NFRAMES
TRAIN_PER = 0.8
LR = 0.01
nb_pool = 2
BATCH_SIZE = 32
DROPOUT = 0.25
DROPOUT2 = 0.5
EPOCHS = 50
FINETUNE_EPOCHS = 10
activation_func2 = 'tanh'


respath = 'results/'
weight_path = join(respath,'weights/')
datapath = '../sdata/s2_audvid'

VIDFILE_NAME = 'viddata_faces2_997.npy'
AUDFILE_NAME = 'auddata_997.npy'

def build_fulllstm_nonconv_model(net_out=18*75):
    model = Sequential()
    model.add(TimeDistributed(Flatten(),input_shape=(75,1,70, 100)))
    model.add(LSTM(netout, name = 'LSTM',input_shape=(75,1,FRAME_ROWS, FRAME_COLS)))
    return(model)

def build_fulllstm_model(net_out=18*75,FRAME_ROWS=128,FRAME_COLS=128):
    model = Sequential()
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv1'),input_shape=(75,1,FRAME_ROWS, FRAME_COLS)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv2')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv3')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same', init='he_normal',name='conv4')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same', init='he_normal',name='conv5')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv6')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv7')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv8')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv9')))
    model.add(BatchNormalization())
    model.add(Activation(activation_func2))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT2))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(512, name = 'LSTM'))
    model.add(Dense(512, init='he_normal',name = 'dense1'))
    model.add(BatchNormalization())
    model.add(Activation(activation_func2))
    model.add(Dropout(DROPOUT2))
    model.add(Dense(512, init='he_normal', name = 'dense2'))
    model.add(BatchNormalization())
    model.add(Dense(net_out))
    return model

def build_nonstlstm_model(net_out,FRAME_ROWS=128,FRAME_COLS=128):
    model = Sequential()
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv1'),input_shape=(CHANNELS,1,FRAME_ROWS, FRAME_COLS)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv2')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv3')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same', init='he_normal',name='conv4')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same', init='he_normal',name='conv5')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv6')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv7')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv8')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv9')))
    model.add(BatchNormalization())
    model.add(Activation(activation_func2))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT2))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(512, name = 'LSTM'))
    model.add(Dense(512, init='he_normal',name = 'dense1'))
    model.add(BatchNormalization())
    model.add(Activation(activation_func2))
    model.add(Dropout(DROPOUT2))
    model.add(Dense(512, init='he_normal', name = 'dense2'))
    model.add(BatchNormalization())
    model.add(Dense(net_out))
    return model

def build_stlstm_model(net_out):
    model = Sequential()
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv1'),batch_input_shape=(32,CHANNELS,1,FRAME_ROWS, FRAME_COLS)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv2')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv3')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same', init='he_normal',name='conv4')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same', init='he_normal',name='conv5')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv6')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv7')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv8')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv9')))
    model.add(BatchNormalization())
    model.add(Activation(activation_func2))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT2))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(512, name = 'LSTM',stateful = True))
    model.add(Dense(512, init='he_normal',name = 'dense1'))
    model.add(BatchNormalization())
    model.add(Activation(activation_func2))
    model.add(Dropout(DROPOUT2))
    model.add(Dense(512, init='he_normal', name = 'dense2'))
    model.add(BatchNormalization())
    model.add(Dense(net_out))
    return model

def build_bidilstm_model(net_out):
    model = Sequential()
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv1'),input_shape=(CHANNELS,1,FRAME_ROWS, FRAME_COLS)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv2')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',name='conv3')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same', init='he_normal',name='conv4')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same', init='he_normal',name='conv5')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv6')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv7')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv8')))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(TimeDistributed(Convolution2D(128, 3, 3, border_mode='same', init='he_normal',name='conv9')))
    model.add(BatchNormalization())
    model.add(Activation(activation_func2))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(nb_pool, nb_pool))))
    model.add(Dropout(DROPOUT2))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(512, name = 'LSTM',stateful = False)))
    model.add(Dense(512, init='he_normal',name = 'dense1'))
    model.add(BatchNormalization())
    model.add(Activation(activation_func2))
    model.add(Dropout(DROPOUT2))
    model.add(Dense(512, init='he_normal', name = 'dense2'))
    model.add(BatchNormalization())
    model.add(Dense(net_out))
    return model

# model = build_nonstlstm_model(18)
# modelst = build_stlstm_model(18)
# modelbi = build_bidilstm_model(18)
# print(modelbi.summary())
