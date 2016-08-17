'''
SUMMARY:  cifar-10 classification, using mlp, 
          te_err=54% after 5 epochs (Still underfitting)
          2 s/epoch on Tesla 2090
AUTHOR:   Qiuqiang Kong
Created:  2016.07.25
Modified: 2016.08.16 Update
--------------------------------------
'''
import numpy as np
import os
import pickle
np.random.seed(1515)
from hat.models import Sequential
from hat.layers.core import InputLayer, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Rmsprop, Adam
import hat.backend as K
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
seq = Sequential()
seq.add( InputLayer( n_in ) )
seq.add( Dense( n_hid, act='relu' ) )
seq.add( Dropout( p_drop=0.2 ) )
seq.add( Dense( n_hid, act='relu' ) )
seq.add( Dropout( p_drop=0.2 ) )
seq.add( Dense( n_out, act='softmax' ) )
md = seq.combine()

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
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, batch_size=500, metrics=['categorical_error'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epochs=20, loss_func='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks, memory_mode=1 )

### predict using model
pred_y = md.predict( te_X, batch_size=100 )
