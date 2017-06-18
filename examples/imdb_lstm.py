"""
SUMMARY:  Example for imdb classification, using LSTM
          Training time: 166 s/epoch. (Titanx)
          Train err: 10%, test error: 18% after 2 epoches. (Better results can be got by tuning hyper-params)
AUTHOR:   Qiuqiang Kong
Created:  2016.05.30
Modified: 2017.06.18
--------------------------------------
"""
import sys
sys.path.append("/user/HS229/qk00006/my_code2015.5-/python/Hat")
import os
import numpy as np
np.random.seed(1337)
from hat.models import Model
from hat.layers.core import InputLayer, Dense, Flatten, Lambda
from hat.layers.embeddings import Embedding
from hat.layers.rnn import SimpleRNN, LSTM, GRU
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical, pad_trunc_seqs
from hat.optimizers import Rmsprop
import hat.backend as K
import pickle
from utils.load_data import load_imdb


# hyper-params
n_words=20000       # regard vocabulary larger than this as unknown
n_proj = 128        # embedding dim
n_hid = 128         # LSTM hid units
max_len = 80        # truncate or pad sentence to this length
n_out = 2

### prepare data
# load data (20000 for training, 5000 for testing)
tr_x, tr_y, te_x, te_y = load_imdb(nb_words=n_words)

# pad & truncate sequences
(tr_x, tr_mask) = pad_trunc_seqs(tr_x, max_len, pad_type='pre')
(te_x, te_mask) = pad_trunc_seqs(te_x, max_len, pad_type='pre')

# sparse target to categorical target
tr_y = sparse_to_categorical(tr_y, n_out)
te_y = sparse_to_categorical(te_y, n_out)

### build model
def mean_pool(input):
    return K.mean(input, axis=1)

lay_in = InputLayer(in_shape=(max_len,))
a = Embedding(n_words, n_proj)(lay_in)
a = LSTM(n_out=n_hid, act='tanh', return_sequences=True)(a)
a = LSTM(n_out=n_hid, act='tanh', return_sequences=True)(a)
a = Lambda(mean_pool)(a)
lay_out = Dense(n_out=n_out, act='softmax')(a)

md = Model(in_layers=[lay_in], out_layers=[lay_out])
md.compile()

# print summary info of model
md.summary()

### callbacks (optional)
# save model every n epoch (optional)
dump_fd = 'imdb_lstm_models'
if not os.path.exists(dump_fd): os.makedirs(dump_fd)
save_model = SaveModel(dump_fd=dump_fd, call_freq=2)

# validate model every n epoch (optional)
validation = Validation(tr_x=tr_x, tr_y=tr_y, 
                        va_x=None, va_y=None, 
                        te_x=te_x, te_y=te_y, 
                        batch_size=500, 
                        metrics=['categorical_error', 'categorical_crossentropy'], 
                        call_freq=1)

# callbacks function
callbacks = [validation, save_model]

### train model
# optimization method
optimizer = Rmsprop(0.001)

md.fit(x=tr_x, y=tr_y, batch_size=32, n_epochs=20, 
       loss_func='categorical_crossentropy', optimizer=optimizer, 
       callbacks=callbacks)

### predict using model
pred_y = md.predict(te_x)