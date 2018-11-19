import numpy as np
import sys
import cv2
from os import listdir
from os.path import isfile, join
import scipy.io
import audio_tools as aud
import skvideo.io
import audio_read as ar
import scipy.io.wavfile as wav
import math
import matplotlib
import matplotlib.pyplot as plt
from skimage import transform as tf
import gen_samples as gs
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

SR = 8000
FPS = 25
SPF = int(SR/FPS)
FRAME_ROWS = 128
FRAME_COLS = 128
NFRAMES = 11 # size of input volume of frames
MARGIN = NFRAMES/2
COLORS = 1 # grayscale
CHANNELS = COLORS*NFRAMES
OVERLAP = 1.0/2
LPC_ORDER = 8
NET_OUT = int(1/OVERLAP)*(LPC_ORDER+1)
SEQ_LEN = 75
SAMPLE_LEN = SEQ_LEN-2*MARGIN
MAX_FRAMES = 1000*SAMPLE_LEN

MPATH = '/data/dccc/sdata/s2_mat/'
datapath = '/data/dccc/sdata/s2_audvid/'
savepath = '/data/dccc/data/datasets/'

VIDDATA_FACES_PATH='viddata_faces2_chan11_997.npy'
AUDDATA_977_PATH ='auddata_997.npy'
FACE_MARGIN = 20
DEFAULT_FACE_DIM = 292
CASC_PATH = 'haarcascade_frontalface_alt.xml'

meanmat = np.load(MPATH+'MEANMAT_977.npy') # mean tracking points for s2 data
# facePntsIDs =range(0,16).append(27) # ids for face points
mouthPntsIDs = range(49, 60)  # ids for mouth pnts in landmark pnts
sideMouthPntsIDs = [48,54] # side of mouth
facePntsIDs = [36,39,27,42,45,33] # sides of left eye, centre face, sides of right eye, nose


def getMouthBasedOnCenterPoint(newLmPnts,mouthPntsIDs, warpFrame, xRight, xLeft, yUp, yBottom, offset):

    # crop mouth image using new aligned landmark points from wraped frame images
    mouthPoints = newLmPnts[mouthPntsIDs,:]
    mouthCenter = np.round(np.mean(mouthPoints,axis=0))
    mouthXmin = int(mouthCenter[0] - xLeft - offset)
    mouthXmax = int(mouthCenter[0] + xRight + offset)
    mouthYmin = int(mouthCenter[1] - yUp - offset)
    mouthYmax = int(mouthCenter[1] + yBottom + offset)

    minImX = 0
    minImY = 0
    maxImY = np.size(warpFrame, 0) - 1
    maxImX = np.size(warpFrame, 1) - 1

    if mouthYmin < minImY:
        mouthYmin = minImY

    if mouthYmax > maxImY:
        mouthYmax = maxImY

    if mouthXmin < minImX:
        mouthXmin = minImX

    if mouthXmax > maxImX:
        mouthXmax = maxImX

    mouthIm = warpFrame[mouthYmin:mouthYmax, mouthXmin:mouthXmax]
    return mouthIm

def find_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

def process_chanelled_video(mpath,vidpath,name,viddata,vidctr,segment='lips',channels=None):
    vpath = vidpath + name
    MARGIN = channels/2
    SAMPLE_LEN = SEQ_LEN - 2*(channels/2)
    CHANNELS = channels
    if segment == 'faces':
        temp_frames = np.zeros((SAMPLE_LEN*CHANNELS,FRAME_ROWS,FRAME_COLS),dtype="uint8")
    elif segment == 'lips':
        temp_frames = np.zeros((SAMPLE_LEN*CHANNELS,70,100),dtype="uint8")
    mats = scipy.io.loadmat(mpath)
    mats = mats['coord']
    cap = skvideo.io.vread(vpath)
    if segment == 'lips':
        stablePntsIDs = sideMouthPntsIDs #taking side points of mount
        centerpoints = mouthPntsIDs
        offset = 0
        xRight = 50
        xLeft = 50
        yUp = 35
        yBottom = 35
    if segment == 'faces':
        stablePntsIDs = facePntsIDs
        centerpoints = np.array([0,16,27]) # center between sides of face and head
        offset = 4
        xRight = 100
        xLeft = 100
        yUp = 40
        yBottom = 40
    for i in range(SEQ_LEN):
        print('Frame Number ', i)
        show=False
        frame = cv2.cvtColor(cap[i,:,:,:],cv2.COLOR_BGR2GRAY)
        mat = mats[i,:,:]
        lmPnts = mat
        df = pd.DataFrame(mat)
        if df.isnull().values.sum():
            print('No Tracking Points Empty for ', name, ' frame '+ str(i))
            lmPnts = meanmat
            show=True
        mfPnts = meanmat
        src = lmPnts[stablePntsIDs,:]
        dst = mfPnts[stablePntsIDs,:]
        src = np.vstack((src,np.mean(lmPnts[mouthPntsIDs,:],axis=0)))
        dst = np.vstack((dst,np.mean(mfPnts[mouthPntsIDs,:],axis=0)))
        tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
        newLmPnts = tform(lmPnts)   # the aligned landmark points
        warpFrame = tf.warp(frame,inverse_map=tform.inverse)  # wrap the frame image
        #print('warpframe before normalised', warpFrame)

        warpFrame = warpFrame * 255.0  # note output from wrap is double image (value range [0,1])
        warpFrame = warpFrame.astype('uint8')
        #print('warpframe', warpFrame)
        croppedImg = getMouthBasedOnCenterPoint(newLmPnts,centerpoints, warpFrame, xRight, xLeft, yUp, yBottom, offset)
        #print('cropped Image, dimensions', croppedImg, croppedImg.shape)
        if segment == 'faces':
            face = cv2.resize(croppedImg,(FRAME_COLS,FRAME_ROWS))
        elif segment == 'lips':
            face = croppedImg
        if show:
            np.save('tempface_'+name+'_frame_'+str(i)+'.npy',face)
        face = np.expand_dims(face,axis=2)
        face = face.transpose(2,0,1)
        temp_frames[i*COLORS:i*COLORS+COLORS,:,:] = face
    for i in np.arange(MARGIN,SAMPLE_LEN+MARGIN):
        viddata[vidctr,:,:,:] = temp_frames[COLORS*(i-MARGIN):COLORS*(i+MARGIN+1),:,:]
        vidctr = vidctr+1
    return vidctr

def samp(path,SR=8000,SAMP_SKIP=50/8):
    swave = wav.read(path)
    y = np.zeros(8000*3,dtype="float32")
    for j in range(SR*3):
        y[j]= swave[1][j*(SAMP_SKIP)]
    return(norm(y))

def process_chanelled_audio(audpath,name, auddata, audctr,channels=None):
    SEQ_LEN = 75
    SAMPLE_LEN =75
    MARGIN = 0
    apath = audpath+name
    if channels != 75:
        MARGIN = channels/2
        SAMPLE_LEN = SEQ_LEN - 2*(channels/2)
    # audio processing
    (y,sr) = ar.audio_read(apath,sr=SR)
    win_length = SPF
    hop_length = int(SPF*OVERLAP)
    [a,g,e] = aud.lpc_analysis(y,LPC_ORDER,window_step=hop_length,window_size=win_length)
    lsf = aud.lpc_to_lsf(a)
    #print('lsf dimensions ', lsf.shape)
    lsf = lsf[(MARGIN)*int(1/OVERLAP):(SAMPLE_LEN+MARGIN)*int(1/OVERLAP),:]
    lsf_concat = np.concatenate((lsf[::2,:],lsf[1::2,:]),axis=1) # MAGIC NUMBERS for half overlap
    g = g[(MARGIN)*int(1/OVERLAP):(SAMPLE_LEN+MARGIN)*int(1/OVERLAP),:]
    g_concat = np.concatenate((g[::2,:],g[1::2,:]),axis=1) # MAGIC NUMBERS for half overlap
    feat = np.concatenate((lsf_concat,g_concat),axis=1)
    auddata[audctr*SAMPLE_LEN:(audctr*SAMPLE_LEN+SAMPLE_LEN),:] = feat
    audctr = audctr+1
    return audctr

def show_progress(progress, step):
    progress += step
    sys.stdout.write("Processing progress: %d%%	\r"%(int(progress)))
    sys.stdout.flush()
    return progress

def norm(x):
    return(x/float(np.amax(x)))

def genfunc(Input_coeff, aud_offset=27*2,STEP=68, plot=False):
    Y_pred=Input_coeff
    i=0
    lsf,g = gs.get_lsf(Y_pred)
    lpc_i = aud.lsf_to_lpc(lsf)
    g_i = g
    x = gs.synthesize(lpc_i,g_i)
    #diff = 24000-len(x)
    print('length of x',len(x)) # 10880
    if plot:
        plt.plot(norm(x))
        plt.title('Reconstructed Waveform')
        plt.xlabel('Time Steps')
        plt.ylabel('Amplitude')
        plt.show()
    return(norm(x))
