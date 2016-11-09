'''
SUMMARY:  Calculate features for chn0, chn1, mix respectively
AUTHOR:   Qiuqiang Kong
Created:  2016.09.23 This file is modified from mir-1k_source_separation
Modified: -
--------------------------------------
'''
import numpy as np
import os
from scipy import signal
import cPickle
import config as cfg
import wavio
import matplotlib.pyplot as plt
from hat.preprocessing import mat_2d_to_3d, pad_trunc_seq


### Extract features
# readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

# calculate complex spectrogram
def GetSpectrogram( x ):
    win = np.ones(1024)/1024.
    [f, t, X] = signal.spectral.spectrogram( x, window=win, nperseg=1024, noverlap=0, \
                    detrend=False, return_onesided=True, mode='complex' )
                    
    X = X.T		# size: N*(nperseg/2+1)
    return X

# calculate all features
def CalculateAllSpectrogram( wavs_fd, fe_fd ):
    names = os.listdir( wavs_fd )
    names = sorted( names )
    cnt = 0
    for na in names:      
        print cnt, na
        path = wavs_fd + '/' + na
        data, fs = readwav( path )
        chn0 = data[:,0]
        chn1 = data[:,1]
        mix = np.mean( data, axis=1 )        

        # get chn0, chn1, mix complex spectrogram respectively
        X_chn0 = GetSpectrogram( chn0 )
        X_chn1 = GetSpectrogram( chn1 )
        X_mix = GetSpectrogram( mix )
        
        # dump
        cPickle.dump( X_chn0, open( fe_fd+'/chn0/'+na[0:-4]+'.p', 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
        cPickle.dump( X_chn1, open( fe_fd+'/chn1/'+na[0:-4]+'.p', 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
        cPickle.dump( X_mix, open( fe_fd+'/mix/'+na[0:-4]+'.p', 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )

        cnt += 1

### Prepare data
# check if na in na_list
def na_in_na_list( na, na_list ):
    for e in na_list:
        if na.startswith( e ):
            return True
    return False
           
# pad matrix with zero
def pad_zero_pre( X, num ):
    pad = np.zeros( (num, X.shape[1]) )
    return np.concatenate( (pad, X) )

# load data
def LoadData( fe_fd, agg_num, hop, na_list ):
    mix_names = sorted( os.listdir( fe_fd + '/mix' ) )
    chn0_names = sorted( os.listdir( fe_fd + '/chn0' ) )
    chn1_names = sorted( os.listdir( fe_fd + '/chn1' ) )
    
    
    X2d_list = []
    X3d_list = []
    for na in mix_names:
        if na_in_na_list( na, na_list ):
            X = cPickle.load( open( fe_fd + '/mix/' + na, 'rb' ) )
            X2d_list.append( X )
            X_pad = pad_zero_pre( X, agg_num-1 )
            X3d = mat_2d_to_3d( X_pad, agg_num, hop )
            X3d_list.append( X3d )

    y2d_chn0_list = []
    y3d_chn0_list = []
    for na in chn0_names:
        if na_in_na_list( na, na_list ):
            X = cPickle.load( open( fe_fd + '/chn0/' + na, 'rb' ) )
            y2d_chn0_list.append( X )
            X_pad = pad_zero_pre( X, agg_num-1 )
            y3d = mat_2d_to_3d( X_pad, agg_num, hop )
            y3d_chn0_list.append( y3d )
            
    
    y2d_chn1_list = []
    y3d_chn1_list = []
    for na in chn1_names:
        if na_in_na_list( na, na_list ):
            X = cPickle.load( open( fe_fd + '/chn1/' + na, 'rb' ) )
            y2d_chn1_list.append( X )
            X_pad = pad_zero_pre( X, agg_num-1 )
            y3d = mat_2d_to_3d( X_pad, agg_num, hop )
            y3d_chn1_list.append( y3d )
        
    X2d = np.concatenate( X2d_list, axis=0 )                # shape: (n_songs*n_chunks, n_freq)
    X3d = np.concatenate( X3d_list, axis=0 )                # shape: (n_songs*n_chunks, n_time, n_freq)
    y2d_chn0 = np.concatenate( y2d_chn0_list, axis=0 )      # shape: (n_songs*n_chunks, n_freq)
    y2d_chn1 = np.concatenate( y2d_chn1_list, axis=0 )      # shape: (n_songs*n_chunks, n_freq)
    y3d_chn0 = np.concatenate( y3d_chn0_list, axis=0 )      # shape: (n_songs*n_chunks, n_time, n_freq)
    y3d_chn1 = np.concatenate( y3d_chn1_list, axis=0 )      # shape: (n_songs*n_chunks, n_time, n_freq)
    
    return X2d, X3d, y2d_chn0, y2d_chn1, y3d_chn0, y3d_chn1

    
### Recover wav
# recover pred spectrogram's phase from ground truth's phase
def real_to_complex( out_X, gt_X ):
    theta = np.angle( gt_X )
    cmplx = out_X * np.cos( theta ) + out_X * np.sin( theta ) * 1j
    return cmplx
    
# recover whole spectrogram from half spectrogram
def half_to_whole( X ):
    return np.hstack( ( X, np.fliplr( np.conj( X[:,1:-1] ) ) ) )

# recover wav from whole spectrogram
def ifft_to_wav( X ):
    return np.real( np.fft.ifft( X ).flatten() )
    
# recover wav from abs spectrogram
def recover_wav_from_abs( X, gt_X ):    
    X = real_to_complex( X, gt_X )
    X = half_to_whole( X )
    s = ifft_to_wav( X )
    return s

# recover wav from complex spectrogram
def recover_wav_from_cmplx( X ):    
    X = half_to_whole( X )
    s = ifft_to_wav( X )
    return s
    
    
### Write out wav
def write_wav( x, fs, path ):
    scaled = np.int16( x/np.max(np.abs(x)) * 16384. )
    wavio.write( path, scaled, fs, sampwidth=2)
    
    
### Main function
if __name__ == '__main__':
    if not os.path.exists( cfg.fe_fd ): os.makedirs( cfg.fe_fd )
    if not os.path.exists( cfg.fe_fft_fd ): os.makedirs( cfg.fe_fft_fd )
    if not os.path.exists( cfg.fe_fft_fd + '/mix' ): os.makedirs( cfg.fe_fft_fd + '/mix' )
    if not os.path.exists( cfg.fe_fft_fd + '/chn0' ): os.makedirs( cfg.fe_fft_fd + '/chn0' )
    if not os.path.exists( cfg.fe_fft_fd + '/chn1' ): os.makedirs( cfg.fe_fft_fd + '/chn1' )
    if not os.path.exists( cfg.results_fd ): os.makedirs( cfg.results_fd )
    CalculateAllSpectrogram( cfg.wavs_fd, cfg.fe_fft_fd )