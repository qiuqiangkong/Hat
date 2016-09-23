'''
SUMMARY:  Example for imdb classification, using LSTM
          Training time: 35 s/epoch. (TitanX)
          Train err: 15%, test error: 20% after 2 epoches. (Better results can be got by tuning hyper-params)
AUTHOR:   Qiuqiang Kong
Created:  2016.05.30
Modified: -
--------------------------------------
'''
import os
import numpy as np
np.random.seed(1337)
from hat.models import Sequential
from hat.layers.core import InputLayer, Dense, Flatten, Lambda
from hat.layers.embeddings import Embedding
from hat.layers.rnn import SimpleRnn, LSTM, GRU
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical, pad_trunc_seqs
from hat.optimizers import Rmsprop
import hat.backend as K
import prepare_data as pp_data
import pickle


# hyper-params
n_words=20000       # regard vocabulary larger than this as unknown
n_proj = 128        # embedding dim
n_hid = 128         # LSTM hid units
max_len = 80        # truncate or pad sentence to this length
n_out = 2

### prepare data
# load data (20000 for training, 5000 for testing)
tr_X, tr_y, te_X, te_y = pp_data.load_data( nb_words=n_words )

# pad & truncate sequences
tr_X = pad_trunc_seqs( tr_X, max_len, pad_type='pre' )
te_X = pad_trunc_seqs( te_X, max_len, pad_type='pre' )

# sparse target to categorical target
tr_y = sparse_to_categorical( tr_y, n_out )
te_y = sparse_to_categorical( te_y, n_out )

### build model
def mean_pool( input ):
    return K.mean( input, axis=1 )

seq = Sequential()
seq.add( InputLayer( max_len ) )
seq.add( Embedding( n_words, n_proj ) )
seq.add( LSTM( n_hid, 'tanh', return_sequence=True ) )
seq.add( Lambda( mean_pool ) )
seq.add( Dense( n_out, 'softmax' ) )
md = seq.combine()

# print summary info of model
md.summary()

### optimization method
optimizer = Rmsprop(0.001)

### callbacks (optional)
# save model every n epoch (optional)
if not os.path.exists('Md'): os.makedirs('Md')      # create folder
save_model = SaveModel( dump_fd='Md', call_freq=2 )

# validate model every n epoch (optional)
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, metrics=['categorical_error', 'categorical_crossentropy'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=32, n_epochs=20, loss_func='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

### predict using model
pred_y = md.predict( te_X )
