"""
SUMMARY:  Example for mnist classification, using CNN. 
          Training time: 18 s/epoch. (TitanX GPU)
          Test error: 0.85% after 10 epoches. (Still underfitting. Better results can be obtained by tuning hyper-params)
Ref:      https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
AUTHOR:   Qiuqiang Kong
Created:  2016.05.18
Modified: 2016.07.26 Make code clearer
          2016.08.16 Update
--------------------------------------
"""
import numpy as np
np.random.seed(1515)
import os
from hat.models import Model
from hat.layers.core import InputLayer, Dense, Dropout, Activation, Flatten
from hat.layers.cnn import Conv2D
from hat.layers.normalization import BN
from hat.layers.pooling import MaxPooling2D
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Adam
from hat import serializations
import hat.backend as K
from utils.load_data import load_mnist

# resize data for fit into CNN. shape: (n_samples, color_maps, height, weight)
def reshape(x):
    N = len(x)
    return x.reshape((N, 1, 28, 28))

# load & prepare data
tr_x, tr_y, va_x, va_y, te_x, te_y = load_mnist()
tr_x = reshape(tr_x)
va_x = reshape(va_x)
te_x = reshape(te_x)

# init params
n_in = 784
n_hid = 500
n_out = 10

# sparse label to 1-of-K categorical label
tr_y = sparse_to_categorical(tr_y, n_out)
va_y = sparse_to_categorical(va_y, n_out)
te_y = sparse_to_categorical(te_y, n_out)

### Build model
lay_in = InputLayer(in_shape=(1, 28, 28))

a = Conv2D(n_outfmaps=32, n_row=3, n_col=3, act='linear', strides=(1, 1), border_mode=(1, 1))(lay_in)
a = BN(axis=(0, 2, 3))(a)
a = Activation('relu')(a)
a = MaxPooling2D(pool_size=(2, 2))(a)
a = Dropout(p_drop=0.2)(a)

a = Conv2D(n_outfmaps=64, n_row=3, n_col=3, act='linear', strides=(1, 1), border_mode=(1, 1))(a)
a = BN(axis=(0, 2, 3))(a)
a = Activation('relu')(a)
a = MaxPooling2D(pool_size=(2, 2))(a)
a = Dropout(p_drop=0.2)(a)

a = Conv2D(n_outfmaps=128, n_row=3, n_col=3, act='linear', strides=(1, 1), border_mode=(1, 1))(a)
a = BN(axis=(0, 2, 3))(a)
a = Activation('relu')(a)
a = MaxPooling2D(pool_size=(2, 2))(a)
a = Dropout(p_drop=0.2)(a)

a = Flatten()(a)
a = Dense(n_out=n_hid, act='linear')(a)
a = BN(axis=0)(a)
a = Activation('relu')(a)
a = Dropout(p_drop=0.2)(a)

lay_out = Dense(n_out=n_out, act='softmax')(a)

md = Model(in_layers=[lay_in], out_layers=[lay_out])
md.compile()

# print summary info of model
md.summary()

### callbacks (optional)
# save model every n epoch (optional)
dump_fd = 'mnist_cnn_models'
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