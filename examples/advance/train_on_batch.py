"""
SUMMARY:  Example of training on batch, data generator, balanced data generator. 
AUTHOR:   Qiuqiang Kong
Created:  2016.06.18
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
import hat.backend as K

file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.dirname(os.path.dirname(file_path)))
from utils.load_data import load_mnist


class DataGenerator(object):
    def __init__(self, batch_size, type):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        
    def generate(self, xs, ys):
        x = xs[0]
        y = ys[0]
        batch_size = self._batch_size_
        (n_samples, n_features) = x.shape
        
        index = np.arange(n_samples)
        
        if self._type_ == 'test':
            max_epoch = 1
        elif self._type_ == 'train':
            max_epoch = 99999
            np.random.shuffle(index)
        
        pointer = 0
        epoch = 1
        while True:
            batch_x = []
            batch_y = []
            
            if pointer >= n_samples:
                if epoch == max_epoch: break
                epoch += 1
                pointer = 0
                np.random.shuffle(index)
 
            batch_idx = index[pointer : min(pointer + batch_size, n_samples)]
            pointer += batch_size
            yield x[batch_idx], y[batch_idx]
        
        
class BalanceDataGenerator(object):
    def __init__(self, batch_size, type):
        assert type in ['train']
        self._batch_size_ = batch_size
        self._type_ = type
        
    def generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_features) = x.shape
        (n_samples, n_labs) = y.shape
        n_each = batch_size // n_labs   
        
        index_list = []
        for i1 in xrange(n_labs):
            index_list.append(np.where(y[:, i1] == 1)[0])
            
        for i1 in xrange(n_labs):
            np.random.shuffle(index_list[i1])
        
        pointer_list = [0] * n_labs
        epoch = 1
        while True:
            batch_x = []
            batch_y = []
            for i1 in xrange(n_labs):
                idx_num = len(index_list[i1])
                if pointer_list[i1] >= idx_num:
                    pointer_list[i1] = 0
                    np.random.shuffle(index_list[i1])
                
                batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_each, idx_num)]
                batch_x.append(x[batch_idx])
                batch_y.append(y[batch_idx])
                pointer_list[i1] += n_each
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y
        
        
# evaluate on batch
def eval(md, gen, xs, ys):
    pred_all = []
    for (batch_x, batch_y) in gen.generate(xs=xs, ys=ys):
        pred = md.predict(batch_x, batch_size=None)[0]
        pred_all.append(pred)
    pred_all = np.concatenate(pred_all, axis=0)
    
    y = ys[0]
    err = metrics.categorical_error(pred_all, y)
    return err


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

# Build model
lay_in = InputLayer(in_shape=(n_in,))
a = Dense(n_out=n_hid, act='relu')(lay_in)
a = Dropout(p_drop=0.2)(a)
a = Dense(n_out=n_hid, act='relu')(a)
a = Dropout(p_drop=0.2)(a)
lay_out = Dense(n_out=n_out, act='softmax')(a)

md = Model(in_layers=[lay_in], out_layers=[lay_out])
md.compile()
md.summary()

# callbacks
dump_fd = 'train_on_batch_models'
if not os.path.exists(dump_fd): os.makedirs(dump_fd)
save_model = SaveModel(dump_fd=dump_fd, call_freq=200, type='iter')

validation = Validation(tr_x=tr_x, tr_y=tr_y, 
                        va_x=None, va_y=None, 
                        te_x=te_x, te_y=te_y, 
                        batch_size=500, 
                        metrics=['categorical_error'], 
                        call_freq=200, 
                        type='iter')

callbacks = [save_model, validation]

# train
balance = True
if balance:
    tr_gen = BalanceDataGenerator(batch_size=500, type='train')
else:
    tr_gen = DataGenerator(batch_size=500, type='train')
    
eval_gen = DataGenerator(batch_size=500, type='test')

eval_freq = 200
tr_time = time.time()
for (tr_batch_x, tr_batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
    md.train_on_batch(batch_x=tr_batch_x, batch_y=tr_batch_y, 
                      loss_func='categorical_crossentropy', 
                      optimizer=Adam(1e-3), 
                      callbacks=callbacks)
    if md.iter_ % eval_freq == 0:
        print "train time:", time.time() - tr_time, "s\n"
        eval_time = time.time()
        tr_err = eval(md=md, gen=eval_gen, xs=[tr_x], ys=[tr_y])
        te_err = eval(md=md, gen=eval_gen, xs=[te_x], ys=[te_y])
        print "iters", md.iter_, "tr_err:", tr_err
        print "iters", md.iter_, "te_err:", te_err
        print "eval time:", time.time() - eval_time, "s\n"
        tr_time = time.time()
    
