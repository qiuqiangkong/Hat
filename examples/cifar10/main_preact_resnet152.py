'''
SUMMARY:  CIFAR-10 classification, using residual 34 layers cnn network. 
          After 10 epoch, tr_err=12%, te_err=23%
          About 4000 s/epoch in 
Ref:      "Identity mappings in deep residual networks"
AUTHOR:   Qiuqiang Kong
Created:  2016.09.21
Modified: -
--------------------------------------
'''
import numpy as np
import os
import pickle
np.random.seed(1515)
from hat.models import Sequential, Model
from hat import serializations
from hat.layers.core import InputLayer, Dense, Dropout, Flatten, Activation, Lambda
from hat.layers.cnn import Convolution2D
from hat.layers.normalization import BN
from hat.layers.pool import MaxPool2D
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Rmsprop, Adam
import hat.backend as K
import config as cfg
import prepare_data as pp_data

# reshape image from N*1024 to N*3*32*32
def reshape_img_for_cnn( x ):
    N = x.shape[0]
    return np.reshape( x, ( N, 3, 32, 32 ) )

def mean_pool( input ):
    return K.mean( input, axis=(2,3) )

# load data
tr_X, tr_y, te_X, te_y = pp_data.load_data()

# normalize data
scaler = pp_data.get_scaler( tr_X )
tr_X = pp_data.transform( tr_X, scaler )
te_X = pp_data.transform( te_X, scaler )

# reshape X to shape: (n_pictures, n_fmaps=3, n_row=32, n_col=32)
tr_X = reshape_img_for_cnn( tr_X )
te_X = reshape_img_for_cnn( te_X )

# init params
n_out = 10

# sparse label to 1-of-K categorical label
tr_y = sparse_to_categorical(tr_y, n_out)
te_y = sparse_to_categorical(te_y, n_out)
print tr_X.shape
print tr_y.shape

def add_layers( inputs ):
    return inputs[0] +  inputs[1]

def mean_pool( input ):
    return K.mean( input, axis=(2,3) )


### Build model

def add_n_bottleneck_blocks( in_layer, n_outfmaps, n_repeat, is_first_layer ):
    assert n_outfmaps%4==0
    a0 = in_layer
    for i1 in xrange( n_repeat ):
        if i1==0:
            if is_first_layer is True: 
                strides=(1,1)
                shortcut = a0
            else: 
                strides=(2,2)
                shortcut = Convolution2D( n_outfmaps=n_outfmaps, n_row=1, n_col=1, act='linear', 
                                          border_mode='valid', strides=strides )( a0 )
        else:
            strides=(1,1)
            shortcut = a0
            
        a1 = BN(axes=(0,2,3))( a0 )
        a2 = Activation( 'relu' )( a1 )
        a3 = Convolution2D( n_outfmaps=n_outfmaps, n_row=1, n_col=1, act='linear', 
                            border_mode='valid', strides=strides )( a2 )
        a4 = BN(axes=(0,2,3))( a3 )
        a5 = Activation( 'relu' )( a4 )
        a6 = Convolution2D( n_outfmaps=n_outfmaps/4, n_row=3, n_col=3, act='linear', 
                            border_mode=(1,1), strides=(1,1) )( a5 )
        a7 = BN(axes=(0,2,3))( a6 )
        a8 = Activation( 'relu' )( a7 )
        a9 = Convolution2D( n_outfmaps=n_outfmaps, n_row=1, n_col=1, act='linear', 
                            border_mode='valid', strides=(1,1) )( a8 )
    
        a10 = Lambda( add_layers )( [shortcut, a9] )
        
        a0 = a10
        
    return a0



x0= InputLayer( in_shape=(3, 32, 32) )
x1 = Convolution2D( n_outfmaps=64, n_row=3, n_col=3, act='relu', border_mode=(1,1) )(x0)
x2 = add_n_bottleneck_blocks( x1, n_outfmaps=64, n_repeat=16, is_first_layer=True )
x3 = add_n_bottleneck_blocks( x2, n_outfmaps=128, n_repeat=16, is_first_layer=False )
x4 = add_n_bottleneck_blocks( x3, n_outfmaps=256, n_repeat=16, is_first_layer=False )

y1 = Lambda( mean_pool )(x4)
y2 = Flatten()(y1)
y3 = Dense( n_out, act='softmax' )(y2)
md = Model( [x0], [y3] )

# print summary info of model
md.summary()

### optimization method
optimizer = Adam(1e-3)

### callbacks (optional)
# save model every n epoch (optional)
if not os.path.exists('Md'): os.makedirs('Md')      # create folder
save_model = SaveModel( dump_fd='Md', call_freq=1 )

# validate model every n epoch (optional)
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, batch_size=32, metrics=['categorical_error'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=32, n_epochs=100, loss_func='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks, verbose=2 )

### predict using model
pred_y = md.predict( te_X, batch_size=32 )