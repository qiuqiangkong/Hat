'''
SUMMARY:  Do source separation and recover wav
AUTHOR:   Qiuqiang Kong
Created:  2016.09.28
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
from hat import serializations
import numpy as np
import os
import prepare_data as pp_data
import config as cfg
from main_dnn import mul
from main_rnn import get_last
from mir_eval.separation import bss_eval_sources


n_freq = 513
agg_num = 3     # This value should be the same as the training phase!
hop = 1     # hop must be 1
n_hid = 500

# load data
te_X2d_mix, te_X3d_mix, te_y2d_chn0, te_y2d_chn1, te_y3d_chn0, te_y3d_chn1 = pp_data.LoadData( cfg.fe_fft_fd, agg_num, hop, [cfg.te_list[0]] )

# load model
md = serializations.load( 'Md/md100.p' )

# get predicted abs spectrogram
[out_chn0, out_chn1] = md.predict( np.abs( te_X3d_mix ) )

# recover wav
s_out_chn0 = pp_data.recover_wav_from_abs( out_chn0, te_X2d_mix )
s_out_chn1 = pp_data.recover_wav_from_abs( out_chn1, te_X2d_mix )
s_gt_chn0 = pp_data.recover_wav_from_cmplx( te_y2d_chn0 )
s_gt_chn1 = pp_data.recover_wav_from_cmplx( te_y2d_chn1 )

# write out wavs
pp_data.write_wav( s_out_chn0, 16000., cfg.results_fd + '/' + 'recover_chn0.wav' )
pp_data.write_wav( s_out_chn1, 16000., cfg.results_fd + '/' + 'recover_chn1.wav' )

# evaluate sdr, sir, sar, perm
# chn0
sdr, sir, sar, perm = bss_eval_sources( s_gt_chn0, s_out_chn0 )
print 'chn0: sdr, sir, sar, perm:', sdr, sir, sar, perm

# chn1
sdr, sir, sar, perm = bss_eval_sources( s_gt_chn1, s_out_chn1 )
print 'chn1: sdr, sir, sar, perm:', sdr, sir, sar, perm