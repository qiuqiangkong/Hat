"""
SUMMARY:  Build model with multi inputs and outputs. 
AUTHOR:   Qiuqiang Kong
Created:  2017.06.18
Modified: -
--------------------------------------
"""
import numpy as np
np.random.seed(1515)
import os
import sys
import inspect
import theano
import theano.tensor as T
from hat.models import Model
from hat.layers.core import InputLayer, Dense, Dropout, Lambda
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Adam
from hat import serializations
import hat.objectives as obj
import hat.backend as K

file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.dirname(os.path.dirname(file_path)))
from utils.load_data import load_mnist


def your_loss(md):
    [out_node1, out_node2] = md.out_nodes_
    [gt_node1, gt_node2] = md.gt_nodes_
    loss1 = T.mean(T.nnet.categorical_crossentropy(out_node1, gt_node1))
    loss2 = T.mean(T.sqr(out_node2 - gt_node2))
    return loss1 + loss2
    
def your_metric(md):
    [out_node1, out_node2] = md.out_nodes_
    [gt_node1, gt_node2] = md.gt_nodes_
    out_node = (out_node1 + out_node2) / 2
    gt_node = (gt_node1 + gt_node2) / 2
    return obj.categorical_error(out_node, gt_node)

def merge(inputs, **kwargs):
    [a1, a2] = inputs
    return T.concatenate((a1, a2), axis=-1)

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
lay_in1 = InputLayer(in_shape=(n_in,))
lay_in2 = InputLayer(in_shape=(n_in,))
a1 = Dense(n_out=n_hid, act='relu')(lay_in1)
a2 = Dense(n_out=n_hid, act='relu')(lay_in2)
b = Lambda(merge)([a1, a2])
b = Dense(n_out=n_hid, act='relu')(b)
lay_out1 = Dense(n_out=n_out, act='softmax')(b)
lay_out2 = Dense(n_out=n_out, act='softmax')(b)

md = Model(in_layers=[lay_in1, lay_in2], out_layers=[lay_out1, lay_out2])
md.compile()
md.summary()

# validate model every n epoch (optional)
validation = Validation(tr_x=[tr_x, tr_x], tr_y=[tr_y, tr_y], 
                        va_x=None, va_y=None, 
                        te_x=[te_x, te_x], te_y=[te_y, te_y], 
                        batch_size=500, 
                        metrics=[your_metric], 
                        call_freq=1)

# callbacks function
callbacks = [validation]

### train model
# optimization method
optimizer = Adam(lr=0.001)        # Try SGD, Adagrad, Rmsprop, etc. instead

md.fit(x=[tr_x, tr_x], y=[tr_y, tr_y], batch_size=500, n_epochs=101, 
       loss_func=your_loss, optimizer=optimizer, 
       callbacks=callbacks)