'''
SUMMARY:  Using deep stacked Rnn, Lstm, GRU do char prediction & generation
          Actully this dataset is too small to use long seqs. Hense the advantage of LSTM, GRU is not obvious. 
          Training time: 2.5 s/epoch. (Tesla M2090)
          test crossentropy: 2.44 after 8 epoches (You can easily beat this by tuning hyper-params and structure)
          Ref: https://gist.github.com/karpathy/d4dee566867f8291f086
AUTHOR:   Qiuqiang Kong
Created:  2016.05.18
Modified: 2016.05.25 Modified
--------------------------------------
'''
import sys
sys.path.append('..')
import numpy as np
np.random.seed(1515)
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Dense, Flatten
from Hat.layers.rnn import SimpleRnn, LSTM, GRU
from Hat.callbacks import Callback, Validation
from Hat.preprocessing import sparse_to_categorical
from Hat.optimizers import Rmsprop
import Hat.backend as K
import pickle

# prepare data
def GetXy( data, agg_num ):
    X, y = [], []
    for i1 in xrange( agg_num, len(data) ):
        ids = np.array( [ char_to_ix[ch] for ch in data[i1-agg_num:i1] ] )
        x = sparse_to_categorical( ids, vocab_size )
        X.append(x)
        y.append( sparse_to_categorical( np.array( [ char_to_ix[ data[i1] ] ] ), vocab_size ) )        
    X = np.array( X )
    y = np.array( y )
    y = y.reshape((y.shape[0],y.shape[2]))
    return X, y

# this is an example of building your own callback function 
# This function will be called every n-epoch training finished. 
class GenerateChar( Callback ):
    def __init__( self, vocab_size, ix_to_char, call_freq ):
        self._vocab_size = vocab_size
        self._call_freq = call_freq
        self._ix_to_char = ix_to_char
        
    def compile( self, md ):
        self._md = md
        
    def call( self ):
        x = None
        chs = ''
        N = 200     # num of chars to be generated
        for i1 in xrange( N ):
            ix, x = self._generate(x)
            chs += self._ix_to_char[ix]
        print '\n--------------\n', chs, '\n--------------'
        
    def _generate( self, x ):
        if x is None:
            x = np.zeros( ( agg_num, self._vocab_size ) )
        input = x.reshape( (1,)+x.shape )   # input dim must be 3d, (batch_num=1, n_time, vocab_size)
        p_y_pred = md.predict( input )[0].flatten()
        ix = np.random.choice( self._vocab_size, p=p_y_pred)
        tmp = np.zeros( self._vocab_size )
        tmp[ix] = 1
        x = np.vstack( ( x[1:], tmp ) )
        return ix, x

### hyperparameters
n_hid = 50      # size of hidden layer of neurons
agg_num = 20     # concatenate frames

### load data & preparation
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

tr_data = data[0:-1000]
te_data = data[-1000:]
tr_X, tr_y = GetXy( tr_data, agg_num )
te_X, te_y = GetXy( te_data, agg_num )
print 'shape tr_X:', tr_X.shape, 'shape tr_y:', tr_y.shape

### build model
md = Sequential()
md.add( InputLayer( (agg_num,vocab_size) ) )
md.add( LSTM( n_hid, act='tanh', return_sequence=True ) )      # Try SimpleRnn, LSTM, GRU instead
md.add( LSTM( n_hid, act='tanh', return_sequence=True ) )
md.add( Flatten() )
md.add( Dense( n_out=vocab_size, act='softmax' ) )

md.summary()
md.plot_connection()

# optimizer
optimizer = Rmsprop(0.01)

### validation
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, call_freq=1, 
                        metric_types=['categorical_error', 'categorical_crossentropy'], dump_path='validation.p' )
generate_char = GenerateChar( vocab_size, ix_to_char, call_freq=1 )
callbacks = [ validation, generate_char ]

### train & validate
md.fit( x=tr_X, y=tr_y, batch_size=100, n_epoch=2000, loss_type='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )
