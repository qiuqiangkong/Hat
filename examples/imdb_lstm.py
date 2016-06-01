'''
SUMMARY:  Example for imdb classification, using LSTM
          Training time: 25 s/epoch. (Tesla M2090)
          Test error: 19% after 30 epoches. (Better results can be got by tuning hyper-params)
AUTHOR:   Qiuqiang Kong
Created:  2016.05.30
Modified: -
--------------------------------------
'''
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/Hat')
import numpy as np
np.random.seed(1337)
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Dense, Flatten
from Hat.layers.embeddings import Embedding
from Hat.layers.rnn import SimpleRnn, LSTM, GRU
from Hat.layers.pool import GlobalMeanTimePool
from Hat.callbacks import SaveModel, Validation
from Hat.preprocessing import sparse_to_categorical, pad_trunc_seqs
from Hat.optimizers import Rmsprop
import Hat.backend as K
from Hat.datasets.imdb import load_data
import pickle


# hyper-params
n_words=20000       # regard vocabulary larger than this as unknown
n_proj = 128        # embedding dim
n_hid = 128         # LSTM hid units
max_len = 80        # truncate or pad sentence to this length
n_out = 2

### prepare data
# load data (20000 for training, 5000 for testing)
tr_X, tr_y, te_X, te_y = load_data( nb_words=n_words )

# pad & truncate sequences
tr_X = pad_trunc_seqs( tr_X, max_len, pad_type='pre' )
te_X = pad_trunc_seqs( te_X, max_len, pad_type='pre' )

# sparse target to categorical target
tr_y = sparse_to_categorical( tr_y, n_out )
te_y = sparse_to_categorical( te_y, n_out )

### build model
md = Sequential()
md.add( InputLayer( max_len ) )
md.add( Embedding( n_words, n_proj ) )
md.add( LSTM( n_hid, 'tanh', return_sequence=False ) )
md.add( Dense( n_out, 'softmax' ) )

# print summary info of model
md.summary()
md.plot_connection()

### optimization method
#optimizer = SGD( lr=0.01, rho=0.9 )
optimizer = Rmsprop(0.001)

### callbacks (optional)
# save model every n epoch (optional)
save_model = SaveModel( dump_fd='Md', call_freq=2 )

# validate model every n epoch (optional)
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, metric_types=['categorical_error', 'categorical_crossentropy'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=32, n_epoch=20, loss_type='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )

### predict using model
pred_y = md.predict( te_X )

### save model
md.dump( 'mnist_mlp_md.p' )
