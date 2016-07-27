'''
SUMMARY:  cifar-10 classification, using mlp, 
          te_err=54% after 5 epochs
          2 s/epoch on Tesla 2090
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
from Hat.layers.core import InputLayer, Dense, Dropout
from Hat.callbacks import SaveModel, Validation
from Hat.preprocessing import sparse_to_categorical
from Hat.optimizers import SGD, Rmsprop, Adam
import Hat.backend as K
from prepare_data import load_data

# load data
tr_X, tr_y, te_X, te_y = load_data()

# init params
n_in = 32*32*3
n_hid = 500
n_out = 10

# sparse label to 1-of-K categorical label
tr_y = sparse_to_categorical(tr_y, n_out)
te_y = sparse_to_categorical(te_y, n_out)
print tr_X.shape
print tr_y.shape

### Build model
md = Sequential()
md.add( InputLayer( n_in ) )
md.add( Dense( n_hid, act='relu' ) )
md.add( Dropout( p_drop=0.2 ) )
md.add( Dense( n_hid, act='relu' ) )
md.add( Dropout( p_drop=0.2 ) )
md.add( Dense( n_out, act='softmax' ) )

# print summary info of model
md.summary()
# md.plot_connection()

### optimization method
optimizer = Adam(1e-4)

### callbacks (optional)
# save model every n epoch (optional)
if not os.path.exists('Md'): os.makedirs('Md')      # create folder
save_model = SaveModel( dump_fd='Md', call_freq=2 )

# validate model every n epoch (optional)
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, batch_size=50000, metric_types=['categorical_error'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epoch=20, loss_type='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks, memory_mode=1 )

### predict using model
pred_y = md.predict( te_X, batch_size=100 )
