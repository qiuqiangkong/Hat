'''
SUMMARY:  Example for mnist classification, using MLP
          Training time: 2 s/epoch. (Tesla M2090)
          Test error: 2.10% after 10 epoches. (Still underfitting. Better results can be obtained by tuning hyper-params)
AUTHOR:   Qiuqiang Kong
Created:  2016.05.18
Modified: 2016.05.25 Write annotation
          2016.07.26 Tiny adjust
          2016.08.16 Tiny update
--------------------------------------
'''
import numpy as np
np.random.seed(1515)
import os
from hat.models import Sequential
from hat.layers.core import InputLayer, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Adam
from hat import serializations
import hat.backend as K
from prepare_data import load_data

### load & prepare data
tr_X, tr_y, va_X, va_y, te_X, te_y = load_data()

# init params
n_in = 784
n_hid = 500
n_out = 10

# sparse label to 1-of-K categorical label
tr_y = sparse_to_categorical(tr_y, n_out)
va_y = sparse_to_categorical(va_y, n_out)
te_y = sparse_to_categorical(te_y, n_out)

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
optimizer = Adam( lr=0.001 )        # Try SGD, Adagrad, Rmsprop, etc. instead

### callbacks (optional)
# save model every n epoch (optional)
if not os.path.exists('Md'): os.makedirs('Md')
save_model = SaveModel( dump_fd='Md', call_freq=2 )

# validate model every n epoch (optional)
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, batch_size=500, 
                         metrics=['categorical_error'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epochs=101, loss_func='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

### predict using model
pred_y = md.predict( te_X )


### save & load models
# serializations.save( md, 'model.p' )
# md = serializations.load( 'model.p' )
