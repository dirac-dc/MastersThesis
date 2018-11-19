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
#import stLSTM as ls
import keras
from sklearn.decomposition import PCA
import  tensorflow as tf

from layers.subpixel import SubPixel1D, SubPixel1D_v2
from keras import backend as K
from keras.layers import merge
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal, Orthogonal
import DataProcess_Functions as df
import autoencoder_classes as ip

FRAME_ROWS = 128
FRAME_COLS = 128
LIPS_FRAME_ROWS=70
LIPS_FRAME_COLS=100
NFRAMES = 9
MARGIN = NFRAMES/2
COLORS = 1
CHANNELS = COLORS*NFRAMES
TRAIN_PER = 0.6
lr = 0.0001
nb_pool = 2
BATCH_SIZE = 32
DROPOUT = 0.25
DROPOUT2 = 0.5
EPOCHS = 50
FINETUNE_EPOCHS = 10
activation_func2 = 'tanh'

EXP ='BIDI_LSTM'

respath = 'results/'
weight_path = join(respath,'weights/')
datapath = '/data/dccc/datasets'
datapath = '/home/dirac/ImperialFiles/LSTM_FINAL/results/'


print('collecting data')
aud_train = np.load('data/audio_recons.npy')
aud_target = np.load('data/audio_true.npy')

data = ip.data(features = aud_train,targets = aud_target)

print('Done, shape of feat and test', data.train_sample_num, data.test_sample_num)

def create_model(x,keep_prob):
    X=x
    with tf.name_scope('generator'):
      L = 8
      # dim/layer: 4096, 2048, 1024, 512, 256, 128,  64,  32,
      # n_filters = [  64,  128,  256, 384, 384, 384, 384, 384]
      n_filters = [  128,  256,  512, 512 , 512, 512, 512, 512]
      # n_filters = [  256,  512,  512, 512, 512, 1024, 1024, 1024]
      # n_filtersizes = [129, 65,   33,  17,  9,  9,  9, 9]
      # n_filtersizes = [31, 31,   31,  31,  31,  31,  31, 31]
      n_filtersizes = [65, 33, 17,  9 ,  9,  9,  9, 9]
      downsampling_l = []

      print 'building model...'
      print 'starting shape: ', x.get_shape()
      # downsampling layers
      for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
        with tf.name_scope('downsc_conv%d' % l):
        #   x = (Convolution1D(nb_filter=nf, filter_length=fs,
        #           activation=None, border_mode='same', init='orthogonal',
        #           subsample_length=2))(x)
          x = Convolution1D(filters = nf, kernel_size=fs, strides=2, padding='same',kernel_initializer='orthogonal')(x)
          #if l > 0:x = BatchNormalization(mode=2)(x)
          x = BatchNormalization()(x)
          x = LeakyReLU(0.2)(x)
          print 'D-Block: ',l, x.get_shape()
          downsampling_l.append(x)

      # bottleneck layer
      with tf.name_scope('bottleneck_conv'):
        #   x = (Convolution1D(nb_filter=n_filters[-1], filter_length=n_filtersizes[-1],
        #           activation=None, border_mode='same', init='orthogonal',
        #           subsample_length=2))(x)
          x = Convolution1D(filters = n_filters[-1], kernel_size=n_filtersizes[-1], strides=2, padding='same',kernel_initializer='orthogonal')(x)
          x = tf.nn.dropout(x, keep_prob)
          # x = BatchNormalization(mode=2)(x)
          x = BatchNormalization()(x)
          x = LeakyReLU(0.2)(x)
          print 'M-Block: ',l, x.get_shape()

      # upsampling layers
      for l, nf, fs, l_in in reversed(zip(range(L), n_filters, n_filtersizes, downsampling_l)):
        with tf.name_scope('upsc_conv%d' % l):
          # (-1, n/2, 2f)
        #   x = (Convolution1D(nb_filter=2*nf, filter_length=fs,
        #           activation=None, border_mode='same', init='orthogonal'))(x)
          x = Convolution1D(filters = 2*nf, kernel_size=fs, strides=1, padding='same',kernel_initializer='orthogonal')(x)
          # x = BatchNormalization(mode=2)(x)
          x = BatchNormalization()(x)
          #x = Dropout(p=0.5)(x)
          x = tf.nn.dropout(x, keep_prob)
          x = Activation('relu')(x)
          # (-1, n, f)
          print x.get_shape, 'Before sub pixel'

          x = SubPixel1D(x, r=2)
          # (-1, n, 2f)
          x = merge([x, l_in], mode='concat', concat_axis=-1)
          print 'U-Block: ', x.get_shape()

      # final conv layer
      with tf.name_scope('lastconv'):
        # x = Convolution1D(nb_filter=2, filter_length=9,
        #         activation=None, border_mode='same', init=RandomNormal)(x)
        x = Convolution1D(filters = 2, kernel_size=fs, strides=1, padding='same',kernel_initializer='orthogonal')(x)
        x = SubPixel1D(x, r=2)
        print 'Final Output: ', x.get_shape()

      g = merge([x, X], mode='sum')
    return g

def model_run(sess,step = 'train'):
    with sess.as_default():
        if step=='train':
            start = time.time()
            for i in range(int(data.train_sample_num/BATCH_SIZE)):
                print('data sectioning')
                features, target, count = data.get_input_sample(stage='train',bs=BATCH_SIZE)
                print('data sectioning done, moving on to training')
                #bias = [v for v in tf.global_variables() if v.name == "conv1/bias:0"]
                #y_co,y_te= sess.run([y_conv,y_test],feed_dict={x: features, y_: target})
                #print('y_conv,y_test,bias',y_co[0,:5],y_te[0,:5],bias)
                _,g_step,summary, train_accuracy = sess.run([train_step,global_step,merged,accuracy],feed_dict={x: features, y_: target, keep_prob:0.5})
                print('adding to train writer')
                train_writer.add_summary(summary,count)
                print('Epoch %d, step %d, loss %g' % (epoch, count, train_accuracy))
            end = time.time()
            print('Time taken for Epoch Training (minutes) %d : %d' % (epoch,(end-start)/60))
        if step == 'test':
            start=time.time()
            test_accuracy = 0
            for i in range(int(data.test_sample_num/BATCH_SIZE)):
                features, target, count = data.get_input_sample(stage='test',bs=BATCH_SIZE)
                test_accuracy += sess.run([accuracy],feed_dict={x: features, y_: target,keep_prob:1})
            test_accuracy = test_accuracy/range(int(TRAIN_LEN/BATCH_SIZE))
            print('adding to train writer')
            test_writer.add_summary(summary,count)
            print('Epoch %d, step %d, val_loss %g' % (epoch, count, train_accuracy,test_accuracy))
            end=time.time()
            print('Time taken for Epoch Validation (minutes) %d : %d' % (epoch,(end-start)/60))
    return(None)

global_step = tf.Variable(0, name='global_step', trainable=False)
# train_op = optimizer.minimize(loss, global_step=global_step)

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

def main():
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    keep_prob = tf.placeholder(tf.float32)
    # Create the model
    x = tf.placeholder(tf.float32, [None, 4096,1])
    # Define loss and optimizer
    y_true = tf.placeholder(tf.float32, [None, 4096,1])

    # Build the graph for the deep net
    y_pred = create_model(x,keep_prob)
    #y_test = model(x,reuse=True)
    tf.summary.scalar('learning_rate', lr)
    # cross_entropy = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    #correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    #tf.summary.scalar('accuracy', accuracy)

    sqrt_l2_loss = tf.sqrt(tf.reduce_mean((y_pred-y_true)**2 + 1e-6, axis=[1,2]))
    sqrn_l2_norm = tf.sqrt(tf.reduce_mean(y_true**2, axis=[1,2]))
    snr = 20 * tf.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / tf.log(10.)

    avg_sqrt_l2_loss = tf.reduce_mean(sqrt_l2_loss, axis=0)
    avg_snr = tf.reduce_mean(snr, axis=0)

    # track losses
    tf.summary.scalar('l2_loss', avg_sqrt_l2_loss)
    tf.summary.scalar('snr', avg_snr)

    # save losses into collection
    tf.add_to_collection('losses', avg_sqrt_l2_loss)
    tf.add_to_collection('losses', avg_snr)

    train_step = tf.train.AdamOptimizer(learning_rate =1e-4, beta1 = 0.99, beta2 = 0.999).minimize(avg_sqrt_l2_loss,global_step=global_step)
    #apply gradient method
    #optimizer = tf.train.AdamOptimizer(lr =1e-4, b1 = 0.99, b2 = 0.999) #.minimize(loss,global_step=global_step)
    # gv = optimizer.compute_gradients(avg_sqrt_l2_loss, params)
    # g, v = zip(*gv)
    # grads = [alpha*g for g in grads]
    # gv = zip(grads, params)
    # train_step = optimizer.apply_gradients(gv, global_step=global_step)

    merged = tf.summary.merge_all()

    variable_averages = tf.train.ExponentialMovingAverage(0.9999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # with tf.Session() as tsess:
    #     yt = sess.run([ytest],feed_dict={x: features, y_: target})
    #     print('y_conv,y_test',yt)
    alpha=1.0
    with tf.Session() as sess:
        start = time.time()
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('./train', sess.graph)
        test_writer = tf.summary.FileWriter('./test',sess.graph)
        print('Variable names',[v.name for v in tf.global_variables()])
        for epoch in range(EPOCHS):
            if epoch%10==0 and epoch != 0:
                print('model saving')
                save_path = saver.save(sess, "./saved_models/model.ckpt",global_step=global_step)
                print('model restoring')
                saver.restore(sess, tf.train.latest_checkpoint('./saved_models/'))
                print("Model restored.")
            model_run(sess,step='train')
            model_run(sess,step='test')
        end = time.time()
        print('Time Taken: ', start - end)

if __name__ == '__main__':
    main()
