'''
SUMMARY:  Train the autoencoder
          Without autoencoder: tr_err=5.4%, te_err=5.6% (5 epoch)
          use ae1 to pretrain: tr_err=2.1%, te_err=2.9% (20 epoch ae1, 5 epoch finetune)
          use ae1 & ae2 to pretrain: tr_err=1.8%, te_err=2.9% (20 epoch ae1, 20 epoch ae2, 5 epoch finetune)
AUTHOR:   Qiuqiang Kong
Created:  2016.10.06
Modified: -
--------------------------------------
'''
import numpy as np
np.random.seed(1515)
import os
from hat.models import Sequential, Model
from hat.layers.core import InputLayer, Dense, Dropout
from hat.callbacks import SaveModel, Validation
from hat.preprocessing import sparse_to_categorical
from hat.optimizers import SGD, Adam
import hat.objectives as obj
from hat import serializations
import hat.backend as K
from prepare_data import load_data
import theano.tensor as T


### train the first layer autoencoder
def train_ae1():
    
    # init params
    n_in = 784
    n_hid = 500
    p_sparsity = 0.05
    n_epochs=10
    
    # mean squared error loss
    def ae1_loss_mse( md ):
        a1 = md.out_nodes_[0]
        a2 = md.any_nodes_[0]
        gt = md.gt_nodes_[0]
        return obj.mse(a2, gt)
    
    # sparsity loss
    def ae1_loss_sparsity( md ):
        a1 = md.out_nodes_[0]
        a2 = md.any_nodes_[0]
        gt = md.gt_nodes_[0]
        return obj.kl_divergence( K.mean(a1, axis=0), p_sparsity*T.ones(n_hid) )
        
    # total loss
    def ae1_loss_total( md ):
        return ae1_loss_mse(md) + ae1_loss_sparsity(md)
    
    # load & prepare data
    tr_X, tr_y, va_X, va_y, te_X, te_y = load_data()
    
    # Build model
    lay_in0 = InputLayer( n_in )
    lay_a1 = Dense( n_hid, act='sigmoid', name='a1' )( lay_in0 )
    lay_a2 = Dense( n_in, act='sigmoid', name='a2' )( lay_a1 )
    md = Model( in_layers=[lay_in0], out_layers=[lay_a1], any_layers=[lay_a2] )
    
    # print summary info of model
    md.summary()
    
    # optimization method
    optimizer = Adam( lr=0.001 )
        
    # validate model every n epoch (optional)
    validation = Validation( tr_x=tr_X, tr_y=tr_X, va_x=None, va_y=None, te_x=te_X, te_y=te_X, batch_size=500, 
                            metrics=[ae1_loss_mse, ae1_loss_sparsity], call_freq=1, dump_path='validation.p' )
    
    # callbacks function
    callbacks = [validation]
    
    # train model
    md.fit( x=tr_X, y=tr_X, batch_size=500, n_epochs=n_epochs, loss_func=ae1_loss_total, optimizer=optimizer, callbacks=callbacks )
    
    # save model
    serializations.save( md, 'Results/md_ae1.p' )


### train the second autoencoder
def train_ae2():
    
    # init params
    n_hid = 500
    p_sparsity = 0.05
    n_epochs=10
    
    # mean squared error loss
    def ae2_loss_mse( md ):
        a3 = md.any_nodes_[0]
        gt = md.gt_nodes_[0]
        return obj.mse(a3, gt)
    
    # sparsity loss
    def ae2_loss_sparsity( md ):
        a2 = md.out_nodes_[0]
        gt = md.gt_nodes_[0]
        p_sparsity = 0.05
        return obj.kl_divergence( K.mean(a2, axis=0), p_sparsity*T.ones(500) )
        
    # total loss
    def ae2_loss_total( md ):
        return ae2_loss_mse(md) + ae2_loss_sparsity(md)
    
    
    # load & prepare data
    tr_X, tr_y, va_X, va_y, te_X, te_y = load_data()
    
    # hidden activation
    md_ae1 = serializations.load('Results/md_ae1.p')
    hid_tr_a1 = md_ae1.predict( tr_X )
    hid_te_a1 = md_ae1.predict( te_X )
    
    # Build model
    
    lay_a1 = InputLayer( n_hid )
    lay_a2 = Dense( n_hid, act='sigmoid', name='a1' )( lay_a1 )
    lay_a3 = Dense( n_hid, act='sigmoid', name='a2' )( lay_a1 )
    md = Model( in_layers=[lay_a1], out_layers=[lay_a2], any_layers=[lay_a3] )
    
    # print summary info of model
    md.summary()
    
    # optimization method
    optimizer = Adam( lr=0.001 )
    
    # validate model every n epoch (optional)
    validation = Validation( tr_x=hid_tr_a1, tr_y=hid_tr_a1, va_x=None, va_y=None, te_x=hid_te_a1, te_y=hid_te_a1, batch_size=500, metrics=[ae2_loss_mse, ae2_loss_sparsity], call_freq=1, dump_path='validation.p' )
    
    # callbacks function
    callbacks = [validation]
    
    # train model
    md.fit( x=hid_tr_a1, y=hid_tr_a1, batch_size=500, n_epochs=n_epochs, loss_func=ae2_loss_total, optimizer=optimizer, callbacks=callbacks )

    # save model
    serializations.save( md, 'Results/md_ae2.p' )


### fine-tune the whole neural network
def finetune():
    
    # init params
    n_in = 784
    n_hid = 500
    n_out = 10
    
    # load & prepare data
    tr_X, tr_y, va_X, va_y, te_X, te_y = load_data()
    
    # sparse label to 1-of-K categorical label
    tr_y = sparse_to_categorical(tr_y, n_out)
    va_y = sparse_to_categorical(va_y, n_out)
    te_y = sparse_to_categorical(te_y, n_out)
    
    # load pre-trained parameters
    md_ae1 = serializations.load('Results/md_ae1.p')
    md_ae2 = serializations.load('Results/md_ae2.p')
    lay_a1_ae1 = md_ae1.find_layer( 'a1' )
    lay_a2_ae2 = md_ae2.find_layer( 'a2' )
    W1_init, b1_init = lay_a1_ae1.W_, lay_a1_ae1.b_
    W2_init, b2_init = lay_a2_ae2.W_, lay_a2_ae2.b_
    
    # build model
    seq = Sequential()
    seq.add( InputLayer( n_in ) )
    seq.add( Dense( n_hid, W_init=W1_init, b_init=b1_init, act='sigmoid', name='a1' ) )
    seq.add( Dense( n_hid, W_init=W2_init, b_init=b2_init, act='sigmoid', name='a2' ) )
    seq.add( Dense( n_out, act='softmax', name='a3' ) )
    md = seq.combine()
    
    # print summary info of model
    md.summary()
    
    # optimization method
    optimizer = Adam( lr=0.001 )        # Try SGD, Adagrad, Rmsprop, etc. instead
    
    # validate model every n epoch (optional)
    validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, batch_size=500, 
                            metrics=['categorical_error'], call_freq=1, dump_path='validation.p' )
    
    # callbacks function
    callbacks = [validation]
    
    # train model
    md.fit( x=tr_X, y=tr_y, batch_size=500, n_epochs=10, loss_func='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )


### main function
if __name__ == '__main__':
    if not os.path.exists('Results'): os.makedirs('Results')
    train_ae1()
    train_ae2()
    finetune()