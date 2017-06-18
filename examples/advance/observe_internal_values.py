"""
SUMMARY:  Observe interal values, including foward outputs of layers and backward
          gradient of layers. 
AUTHOR:   Qiuqiang Kong
Created:  2017.06.18
Modified: -
--------------------------------------
"""
import numpy as np
np.random.seed(1515)
import os
import sys
import time
import inspect
from hat.models import Model
from hat.layers.core import InputLayer, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Adam
from hat import serializations
from hat import metrics
import hat.objectives as obj
import hat.backend as K

file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.dirname(os.path.dirname(file_path)))
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
a = Dense(n_out=n_hid, act='relu', name='dense1')(lay_in)
a = Dropout(p_drop=0.2)(a)
a = Dense(n_out=n_hid, act='relu', name='dense2')(a)
a = Dropout(p_drop=0.2)(a)
lay_out = Dense(n_out=n_out, act='softmax')(a)

md = Model(in_layers=[lay_in], out_layers=[lay_out])
md.compile()
md.summary()

# observe forward
observe_nodes = [md.find_layer('dense1').output_, 
                 md.find_layer('dense2').output_]
f_forward = md.get_observe_forward_func(observe_nodes)
print md.run_function(func=f_forward, z=[te_x], batch_size=500, tr_phase=0.)

# observe backward
md.set_gt_nodes(target_dim_list=[2])
loss_node = obj.categorical_crossentropy(md.out_nodes_[0], md.gt_nodes_[0])
gparams = K.grad(loss_node + md.reg_value_, md.params_)
observe_nodes = gparams
f_backward = md.get_observe_backward_func(observe_nodes)
print md.run_function(func=f_backward, z=[te_x, te_y], batch_size=500, tr_phase=0.)