"""
SUMMARY:  Usage of Save & Load model
AUTHOR:   Qiuqiang Kong
Created:  2017.06.18
Modified: -
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


# init params
n_in = 784
n_hid = 500
n_out = 10

lay_in = InputLayer(in_shape=(n_in,))
a = Dense(n_out=n_hid, act='relu')(lay_in)
a = Dropout(p_drop=0.2)(a)
a = Dense(n_out=n_hid, act='relu')(a)
a = Dropout(p_drop=0.2)(a)
lay_out = Dense(n_out=n_out, act='softmax')(a)

md = Model(in_layers=[lay_in], out_layers=[lay_out])
md.compile()
md.summary()

# Save model
md_path = 'model.p'
serializations.save(md=md, path=md_path)

# Load model
md_load = serializations.load(md_path)