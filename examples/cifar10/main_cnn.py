'''
SUMMARY:  cifar-10 classification, using vgg like network. 
          After 10 epoch, te_err=28%. 
          About 5 min in Tesla 2090
AUTHOR:   Qiuqiang Kong
Created:  2016.07.25
Modified: -
--------------------------------------
'''
import numpy as np
import os
import pickle
np.random.seed(1515)
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Dense, Dropout, Flatten
from Hat.layers.cnn import Convolution2D
from Hat.layers.pool import MaxPool2D
from Hat.callbacks import SaveModel, Validation
from Hat.preprocessing import sparse_to_categorical
from Hat.optimizers import SGD, Rmsprop, Adam
import Hat.backend as K
from prepare_data import load_data

# reshape image from N*1024 to N*3*32*32
def reshape_img( x ):
    N = x.shape[0]
    return np.reshape( x, ( N, 3, 32, 32 ) )

# load data
tr_X, tr_y, te_X, te_y = load_data()
tr_X = reshape_img( tr_X )
te_X = reshape_img( te_X )

# init params
n_out = 10

# sparse label to 1-of-K categorical label
tr_y = sparse_to_categorical(tr_y, n_out)
te_y = sparse_to_categorical(te_y, n_out)
print tr_X.shape
print tr_y.shape

### Build model
md = Sequential()
md.add( InputLayer( in_shape=(3, 32, 32) ) )
md.add( Convolution2D( n_outfmaps=64, n_row=3, n_col=3, act='relu' ) )
md.add( Convolution2D( n_outfmaps=64, n_row=3, n_col=3, act='relu' ) )
md.add( MaxPool2D( pool_size=(2,2) ) )
md.add( Convolution2D( n_outfmaps=128, n_row=3, n_col=3, act='relu' ) )
md.add( Convolution2D( n_outfmaps=128, n_row=3, n_col=3, act='relu' ) )
md.add( MaxPool2D( pool_size=(2,2) ) )
md.add( Convolution2D( n_outfmaps=256, n_row=3, n_col=3, act='relu' ) )
md.add( Convolution2D( n_outfmaps=256, n_row=3, n_col=3, act='relu' ) )
md.add( MaxPool2D( pool_size=(2,2) ) )
md.add( Flatten() )
md.add( Dense( 500, act='relu' ) )
md.add( Dropout( p_drop=0.2 ) )
md.add( Dense( 500, act='relu' ) )
md.add( Dropout( p_drop=0.2 ) )
md.add( Dense( n_out, act='softmax' ) )


# print summary info of model
md.summary()

### optimization method
optimizer = Adam(1e-4)

### callbacks (optional)
# save model every n epoch (optional)
if not os.path.exists('Md'): os.makedirs('Md')      # create folder
# we do not use save model here because save large model will cause problem. Fixing ..
#save_model = SaveModel( dump_fd='Md', call_freq=2 )

# validate model every n epoch (optional)
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, batch_size=500, metric_types=['categorical_error'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epoch=1, loss_type='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

### predict using model
pred_y = md.predict( te_X, batch_size=500 )
