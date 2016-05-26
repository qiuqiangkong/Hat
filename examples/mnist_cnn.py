'''
SUMMARY:  Example for mnist classification, using Lenet-CNN
          Training time: 25 s/epoch. (Tesla M2090)
          Test error: 0.78% after 20 epoches. (Better results can be got by tuning hyper-params)
Ref:      https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
AUTHOR:   Qiuqiang Kong
Created:  2016.05.18
Modified: -
--------------------------------------
'''
import sys
sys.path.append('..')
import pickle
import numpy as np
np.random.seed(1515)
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Flatten, Dense, Dropout
from Hat.layers.cnn import Convolution2D
from Hat.layers.pool import MaxPool2D
from Hat.callbacks import SaveModel, Validation
from Hat.preprocessing import sparse_to_categorical
from Hat.optimizers import Rmsprop
import Hat.backend as K
from Hat.datasets.mnist import load_data


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
md = Sequential()
md.add( InputLayer( in_shape=(1,28,28) ) )
md.add( Convolution2D( n_outfmaps=32, n_row=3, n_col=3, act='relu') )
md.add( MaxPool2D( pool_size=(2,2) ) )
md.add( Convolution2D( n_outfmaps=32, n_row=3, n_col=3, act = 'relu') )
md.add( MaxPool2D( pool_size=(2,2) ) )
md.add( Dropout( 0.2 ) )
md.add( Flatten() )
md.add( Dense( n_hid, act = 'relu' ) )
md.add( Dropout( 0.5 ) )
md.add( Dense( n_hid, act = 'relu' ) )
md.add( Dense( n_out, act='softmax' ) )

# print summary info of model
md.summary()
md.plot_connection()

# optimization method
optimizer = Rmsprop(0.001)

### callbacks (optional)
# save model every n epoch (optional)
save_model = SaveModel( dump_fd='Md', call_freq=2 )

# validate model every n epoch (optional)
validation = Validation( tr_x=None, tr_y=None, va_x=va_X, va_y=va_y, te_x=te_X, te_y=te_y, metric_types=['categorical_error'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epoch=20, loss_type='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

### predict using model
pred_y = md.predict( te_X )

### save model
md.dump( 'mnist_mlp_md.p' )


'''
# Load model
# Later you can use
md = pickle.load( open( 'Md/md2.p', 'rb' ) )
md.fit(...)     # continue training
md.predict(...) # testing
'''