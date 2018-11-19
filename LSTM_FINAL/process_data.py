import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import sys
import scipy.io
import audio_tools as aud
# import audio_read as ar
#import skvideo.io
import audio_read as ar
import scipy.io.wavfile as wav
import math
import matplotlib.pyplot as plt
import matplotlib
from skimage import transform as tf
import DataProcess_Functions as df
import numpy as np
from numpy import ndarray as nd
SR = 8000
FPS = 25
SPF = int(SR/FPS)
FRAME_ROWS = 128
FRAME_COLS = 128
LIPS_FRAME_ROWS = 70
LIPS_FRAME_COLS = 100
COLORS = 1 # grayscale
OVERLAP = 1.0/2
LPC_ORDER = 8
NET_OUT = int(1/OVERLAP)*(LPC_ORDER+1)
MAX_FRAMES = 1000
CHANNELS = 75

MPATH = '/data/dccc/sdata/s2_mat/'
datapath = '/data/dccc/sdata/s2_audvid/'
savepath = '/data/dccc/datasets/'

viddata_path='viddata_nochann_lips.npy'
auddata_path ='auddata_nochann.npy'



def main():
    print('Creating Data Frames')
    viddata = np.zeros((MAX_FRAMES,CHANNELS,LIPS_FRAME_ROWS,LIPS_FRAME_COLS),dtype="uint8")
    auddata = np.zeros((MAX_FRAMES*75*2,18),dtype="float32")
    print('Data Frames Done')
    vidfiles = [f for f in listdir(datapath) if isfile(join(datapath, f)) and f.endswith(".mpg")]
    np.save(savepath+'audvidorder_s2.npy',vidfiles)
    print('Length of Videos to be processed', len(vidfiles))
    vidctr = 0
    audctr = 0
    progress = 0
    progress_step = 100./len(vidfiles)
    progress = df.show_progress(progress, 0)
    for vf in vidfiles:
        vidctr = df.process_chanelled_video(MPATH+vf.replace('.mpg','.mat'),datapath,vf, viddata, vidctr,channels=CHANNELS)
        audctr = df.process_chanelled_audio(datapath,vf.replace('.mpg','.wav'), auddata, audctr,channels=CHANNELS)
        progress = df.show_progress(progress, progress_step)
    progress = df.show_progress(progress, progress_step)
    assert vidctr==audctr
    print ('Done processing. Saving data to disk...')
    viddata = viddata[:vidctr,:,:,:]
    #auddata = auddata[:audctr,:]
    print('viddata_shape',viddata.shape)
    print('auddata_shape',auddata.shape)
    ########################
    # Saving and Reshaping
    #######################
    if CHANNELS == 75:
        auddata_new = np.zeros((len(vidfiles),18*75),dtype="float32")
        for i in range(len(vidfiles)):
            if (i+1)%75 == 0:
                print('reshaping video', (i+1)/75)
            print(auddata[i*75:(i+1)*75,:].flatten())
            auddata_new[i,:] = auddata[i*75:(i+1)*75,:].flatten()

        viddata.transpose(1,0,2,3)
        print('vid shape',viddata.shape)
        print('aud shape', auddata_new.shape)
        print('Saving Data')
        np.save(savepath+viddata_path,viddata)
        np.save(savepath+auddata_path,auddata_new)
    else:
        print('vid shape',viddata.shape)
        print('aud shape', auddata_new.shape)
        print('Saving Data')
        np.save(savepath+viddata_path,viddata)
        np.save(savepath+auddata_path,auddata_new)

    print ('Done')

if __name__ == "__main__":
    main() # prevents modules from being run in other files
