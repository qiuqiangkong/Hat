'''
SUMMARY:  Example for mnist classification, using Lenet-CNN
          Training time: 18 s/epoch. (Tesla M2090)
          Test error: 1.15% after 10 epoches. (Still underfitting. Better results can be obtained by tuning hyper-params)
Ref:      https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
AUTHOR:   Qiuqiang Kong
Created:  2016.05.18
Modified: 2016.07.26 Make code clearer
          2016.08.16 Update
--------------------------------------
'''
import pickle
import numpy as np
np.random.seed(1515)
import os
from hat.models import Sequential
from hat.layers.core import InputLayer, Flatten, Dense, Dropout
from hat.layers.cnn import Convolution2D
from hat.layers.pool import MaxPool2D
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import Adam
import hat.backend as K
from prepare_data import load_data

# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX( X ):
    N = len(X)
    return X.reshape( (N, 1, 28, 28) )
    
### load & prepare data
tr_X, tr_y, va_X, va_y, te_X, te_y = load_data()
tr_X, va_X, te_X = reshapeX(tr_X), reshapeX(va_X), reshapeX(te_X)

# init params
n_in = 784
n_hid = 500
n_out = 10

# sparse label to 1 of K categorical label
tr_y = sparse_to_categorical(tr_y, n_out)
va_y = sparse_to_categorical(va_y, n_out)
te_y = sparse_to_categorical(te_y, n_out)

### Build model
act = 'relu'
seq = Sequential()
seq.add( InputLayer( in_shape=(1,28,28) ) )
seq.add( Convolution2D( n_outfmaps=32, n_row=3, n_col=3, act='relu') )
seq.add( MaxPool2D( pool_size=(2,2) ) )
seq.add( Convolution2D( n_outfmaps=32, n_row=3, n_col=3, act = 'relu') )
seq.add( MaxPool2D( pool_size=(2,2) ) )
seq.add( Dropout( 0.2 ) )
seq.add( Flatten() )
seq.add( Dense( n_hid, act = 'relu' ) )
seq.add( Dropout( 0.5 ) )
seq.add( Dense( n_hid, act = 'relu' ) )
seq.add( Dense( n_out, act='softmax' ) )
md = seq.combine()

# print summary info of model
md.summary()

# optimization method
optimizer = Adam( lr=0.001 )

### callbacks (optional)
# save model every n epoch (optional)
if not os.path.exists('Md'): os.makedirs('Md')
save_model = SaveModel( dump_fd='Md', call_freq=2 )

# validate model every n epoch (optional)
validation = Validation( tr_x=None, tr_y=None, va_x=va_X, va_y=va_y, te_x=te_X, te_y=te_y, batch_size=500, metrics=['categorical_error'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epochs=50, loss_func='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

### predict using model
pred_y = md.predict( te_X, batch_size=500 )