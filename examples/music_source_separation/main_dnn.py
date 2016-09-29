'''
SUMMARY:  source separation using dnn example
AUTHOR:   Qiuqiang Kong
Created:  2016.09.28
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/user/HS229/qk00006/my_code2015.5-/python/Hat')
from hat.models import Model
from hat.layers.core import InputLayer, Dense, Dropout, Flatten, Lambda
from hat.layers.rnn import SimpleRnn, LSTM, GRU
from hat.optimizers import SGD, Rmsprop, Adam
from hat.callbacks import Validation, SaveModel
import hat.objectives as obj
import numpy as np
import os
import prepare_data as pp_data
import config as cfg

# loss function
def loss_func( out_nodes, any_nodes, gt_nodes ):
    [in0, b1, c1] = any_nodes
    [gt_b, gt_c] = gt_nodes
    return obj.norm_lp( in0*b1, gt_b, 2 ) + obj.norm_lp( in0*c1, gt_c, 2 )

# multipl input spectrogram with mask
def mul( inputs ):
    return inputs[0] * inputs[1][:,-1,:]

n_freq = 513
agg_num = 3
hop = 1     # hop must be 1
n_hid = 500


# train model
def train():
    
    # load data
    tr_X2d_mix, tr_X3d_mix, tr_y2d_chn0, tr_y2d_chn1, tr_y3d_chn0, tr_y3d_chn1 = pp_data.LoadData( cfg.fe_fft_fd, agg_num, hop, na_list=cfg.tr_list )
    
    # build model
    lay_in0 = InputLayer( in_shape=( (agg_num, n_freq) ), name='in1' )
    lay_a0 = Flatten()( lay_in0 )
    lay_a1 = Dense( n_hid, act='relu', name='a1' )( lay_a0 )
    lay_a2 = Dropout( 0.2, name='a2' )( lay_a1 )
    lay_a3 = Dense( n_hid, act='relu', name='a3' )( lay_a2 )
    lay_a4 = Dropout( 0.2, name='a4' )( lay_a3 )
    lay_a5 = Dense( n_hid, act='relu', name='a5' )( lay_a4 )
    lay_a6 = Dropout( 0.2, name='a6' )( lay_a5 )
    lay_b1 = Dense( n_freq, act='sigmoid', name='a7' )( lay_a6 )     # mask_left
    lay_c1 = Dense( n_freq, act='sigmoid', name='a8' )( lay_a6 )     # mask_right
    lay_out_b = Lambda( mul, name='out_b' )( [lay_b1, lay_in0] )
    lay_out_c = Lambda( mul, name='out_c' )( [lay_c1, lay_in0] )
    
    md = Model( in_layers=[lay_in0], out_layers=[lay_out_b, lay_out_c], any_layers=[lay_in0, lay_b1, lay_c1] )
    md.summary()
    
    
    # validation
    validation = Validation( tr_x=[np.abs(tr_y3d_chn0)+np.abs(tr_y3d_chn1)], tr_y=[np.abs(tr_y2d_chn0), np.abs(tr_y2d_chn1)], batch_size=100, metrics=['mse'], call_freq=1, dump_path='validation.p' )
    
    # save model
    if not os.path.exists(cfg.md_fd): os.makedirs(cfg.md_fd)
    save_model = SaveModel( dump_fd=cfg.md_fd, call_freq=2 )
    
    # callbacks
    callbacks = [ validation, save_model ]
    
    # optimizer
    optimizer = Adam(1e-3)
    
    # fit model
    md.fit( [np.abs(tr_y3d_chn0)+np.abs(tr_y3d_chn1)], [np.abs(tr_y2d_chn0), np.abs(tr_y2d_chn1)], \
        batch_size=100, n_epochs=100, loss_func='mse', optimizer=optimizer, callbacks=callbacks, verbose=1 )

if __name__ == '__main__':
    train()