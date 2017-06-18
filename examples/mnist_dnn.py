"""
SUMMARY:  Example for mnist classification, using MLP
          Training time: 2 s/epoch. (Tesla M2090)
          Test error: 2.10% after 10 epoches. (Still underfitting. Better results can be obtained by tuning hyper-params)
AUTHOR:   Qiuqiang Kong
Created:  2016.05.18
Modified: 2017.06.17
--------------------------------------
"""
import numpy as np
np.random.seed(1515)
import os
from hat.models import Model
from hat.layers.core import InputLayer, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Adam
from hat import serializations
import hat.backend as K
from utils.load_data import load_mnist

# load & prepare data
tr_x, tr_y, va_x, va_y, te_x, te_y = load_mnist()

# init params
n_in = 784
n_hid = 500
n_out = 10

# sparse label to 1-of-K categorical label
tr_y = sparse_to_categorical(tr_y, n_out)
va_y = sparse_to_categorical(va_y, n_out)
te_y = sparse_to_categorical(te_y, n_out)

### Build model

lay_in = InputLayer(in_shape=(n_in,))
a = Dense(n_out=n_hid, act='relu')(lay_in)
a = Dropout(p_drop=0.2)(a)
a = Dense(n_out=n_hid, act='relu')(a)
a = Dropout(p_drop=0.2)(a)
lay_out = Dense(n_out=n_out, act='softmax')(a)

md = Model(in_layers=[lay_in], out_layers=[lay_out])
md.compile()

# print summary info of model
md.summary()

### callbacks (optional)
# save model every n epoch (optional)
dump_fd = 'mnist_dnn_models'
if not os.path.exists(dump_fd): os.makedirs(dump_fd)
save_model = SaveModel(dump_fd=dump_fd, call_freq=2)

# validate model every n epoch (optional)
validation = Validation(tr_x=tr_x, tr_y=tr_y, 
                        va_x=None, va_y=None, 
                        te_x=te_x, te_y=te_y, 
                        batch_size=500, 
                        metrics=['categorical_error'], 
                        call_freq=1)

# callbacks function
callbacks = [validation, save_model]

### train model
# optimization method
optimizer = Adam(lr=0.001)        # Try SGD, Adagrad, Rmsprop, etc. instead

md.fit(x=tr_x, y=tr_y, batch_size=500, n_epochs=101, 
       loss_func='categorical_crossentropy', optimizer=optimizer, 
       callbacks=callbacks)

### predict using model
pred_y = md.predict(te_x)