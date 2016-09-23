'''
SUMMARY:  CIFAR-10 classification, using vgg like network. 
          After 10 epoch, tr_err=7.33% te_err=20.95%. 
          About 100 s/epoch in TitanX 2090
Ref:      "Batch normalization: Accelerating deep network training by reducing internal covariate shift."
AUTHOR:   Qiuqiang Kong
Created:  2016.07.25
Modified: -
--------------------------------------
'''
import numpy as np
import os
import pickle
np.random.seed(1515)
from hat.models import Sequential
from hat import serializations
from hat.layers.core import InputLayer, Dense, Dropout, Flatten, Activation
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


### Build model
seq = Sequential()
seq.add( InputLayer( in_shape=(3, 32, 32) ) )

seq.add( Convolution2D( n_outfmaps=64, n_row=3, n_col=3, act='linear', border_mode=(1,1) ) )
seq.add( BN( axes=(0,2,3) ) )
seq.add( Activation('relu') )
seq.add( Convolution2D( n_outfmaps=64, n_row=3, n_col=3, act='linear', border_mode=(1,1) ) )
seq.add( BN( axes=(0,2,3) ) )
seq.add( Activation('relu') )
seq.add( MaxPool2D( pool_size=(2,2) ) )

seq.add( Convolution2D( n_outfmaps=128, n_row=3, n_col=3, act='linear', border_mode=(1,1) ) )
seq.add( BN( axes=(0,2,3) ) )
seq.add( Activation('relu') )
seq.add( Convolution2D( n_outfmaps=128, n_row=3, n_col=3, act='linear', border_mode=(1,1) ) )
seq.add( BN( axes=(0,2,3) ) )
seq.add( Activation('relu') )
seq.add( MaxPool2D( pool_size=(2,2) ) )

seq.add( Convolution2D( n_outfmaps=256, n_row=3, n_col=3, act='linear', border_mode=(1,1) ) )
seq.add( BN( axes=(0,2,3) ) )
seq.add( Activation('relu') )
seq.add( Convolution2D( n_outfmaps=256, n_row=3, n_col=3, act='linear', border_mode=(1,1) ) )
seq.add( BN( axes=(0,2,3) ) )
seq.add( Activation('relu') )
seq.add( MaxPool2D( pool_size=(2,2) ) )

seq.add( Flatten() )
seq.add( Dense( 500, act='linear' ) )
seq.add( BN( axes=0 ) )
seq.add( Activation('relu') )
seq.add( Dense( 500, act='linear' ) )
seq.add( BN( axes=0 ) )
seq.add( Activation('relu') )
seq.add( Dense( n_out, act='softmax' ) )
md = seq.combine()


# print summary info of model
md.summary()

### optimization method
optimizer = Adam(1e-3)

### callbacks (optional)
# save model every n epoch (optional)
if not os.path.exists('Md'): os.makedirs('Md')      # create folder
save_model = SaveModel( dump_fd='Md', call_freq=1 )

# validate model every n epoch (optional)
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, batch_size=1000, metrics=['categorical_error'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epochs=10, loss_func='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

### predict using model
pred_y = md.predict( te_X, batch_size=500 )