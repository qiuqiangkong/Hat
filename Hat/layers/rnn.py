'''
SUMMARY:  Rnn, Lstm
AUTHOR:   Qiuqiang Kong
Created:  2016.05.17
Modified: 2016.05.21 Modify bug in LSTM
--------------------------------------
'''
from core import Layer
from ..globals import new_id
from .. import backend as K
from .. import initializations
from .. import activations
from ..supports import to_list, get_mask
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty


class RnnBase( Layer ):
    __metaclass__ = ABCMeta
    
    def __init__( self, n_out, act, init_type, reg, return_sequence, go_backwards, masking, name ):
        super( RnnBase, self ).__init__( name )
        self._n_out = n_out
        self._act = act
        self._init_type = init_type
        self._reg = reg
        self._return_sequence = return_sequence
        self._go_backwards = go_backwards
        self._masking = masking
    
    @abstractmethod
    def _step( self ):
        pass

    @abstractmethod
    def _scan( self ):
        pass
        
'''
Simple Rnn layer. Using random matrix to init H is better than eye matrix
'''
class SimpleRnn( RnnBase ):
    def __init__( self, n_out, act, init_type='uniform', reg=None, return_sequence=True, go_backwards=False, masking=False, name=None ):
        super( SimpleRnn, self ).__init__( n_out, act, init_type, reg, return_sequence, go_backwards, masking, name )
        
    def __call__( self, in_layers ):
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Rnn can only be one layer!"
        in_layer = in_layers[0]
        
        in_shape = in_layer.out_shape
        input = in_layer.output
        assert len(in_shape)==3, "The dim of input must be 3! Your shape is " + str(in_shape)
        [ batch_size, n_time, n_in ] = in_shape
        
        # reverse data
        if self._go_backwards: input = input[::-1]
        
        # parameters
        self._W = initializations.get( self._init_type )( (n_in, self._n_out), name=str(self._name)+'_W' )
        self._H = initializations.get( self._init_type )( (self._n_out, self._n_out), name=str(self._name)+'_H' )
        #self._H = initializations.get( 'eye' )( self._n_out, name=str(self._name)+'_H' )
        self._b = initializations.get( 'zeros' )( (self._n_out), name=str(self._name)+'_b' )
     
        # scan
        output = self._scan( input )
        
        # mask
        if self._masking==True:
            mask = get_mask( input )
            output *= mask[:,:,None]
        
        # if return_sequence=False only return the last value, size: batch_size*n_in
        if self._return_sequence is False:
            output = output[:,-1,:].flatten(2)
            out_shape = (None, self._n_out)
        else:
            out_shape = (None, n_time, self._n_out)
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = out_shape
        self._output = output
        self._params = [ self._W, self._b, self._H ]
        self._reg_value = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    # size(input): batch_size*n_times*n_in
    def _scan( self, input ):
        assert input.ndim==3
        batch_size, n_times, n_in = input.shape
         
        # to use scan, dimshuffle input to shape: n_times*batch_size*n_in
        results, update = K.scan( self._step, sequences=input.dimshuffle(1,0,2), outputs_info=[K.zeros((batch_size, self._n_out))] )
        
        # dimshuffle output back to shape: batch_size*n_times*n_in
        output = results.dimshuffle(1,0,2)
        return output
    
    # size(x): batch_size*n_in, size(h_): batch_size*n_out
    def _step( self, x, h_ ):
        lin_out = K.dot( x, self._W ) + K.dot( h_, self._H ) + self._b
        h = activations.get( self._act )( lin_out )

        return h
        
    # get regularization
    def _get_reg( self ):
        if self._reg is None:
            _reg_value = 0.
        else:
            _reg_value = self._reg.reg_value( [ self.W, self.H ] )
        return _reg_value
        
class LSTM( RnnBase ):
    def __init__( self, n_out, act, init_type='uniform', reg=None, return_sequence=True, go_backwards=False, masking=False, name=None ):
        super( LSTM, self ).__init__( n_out, act, init_type, reg, return_sequence, go_backwards, masking, name )
        
    def __call__( self, in_layers ):
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Rnn can only be one layer!"
        in_layer = in_layers[0]
        
        # prepare data
        in_shape = in_layer.out_shape
        input = in_layer.output
        assert len(in_shape)==3, "The dim of input must be 3! Your shape is " + str(in_shape)
        [ batch_size, n_time, n_in ] = in_shape
        
        # reverse data
        if self._go_backwards: input = input[::-1]
        
        # parameters
        self._Wg = initializations.get( self._init_type )( (n_in, self._n_out), name=str(self.id)+'_Wg' )
        self._Ug = initializations.get( self._init_type )( (self._n_out, self._n_out), name=str(self.id)+'_Ug' )
        self._bg = initializations.get( 'zeros' )( (self._n_out), name=str(self.id)+'_bg' )
        self._Wi = initializations.get( self._init_type )( (n_in, self._n_out), name=str(self.id)+'_Wi' )
        self._Ui = initializations.get( self._init_type )( (self._n_out, self._n_out), name=str(self.id)+'_Ui' )
        self._bi = initializations.get( 'zeros' )( (self._n_out), name=str(self.id)+'_bi' )
        self._Wf = initializations.get( self._init_type )( (n_in, self._n_out), name=str(self.id)+'_Wf' )
        self._Uf = initializations.get( self._init_type )( (self._n_out, self._n_out), name=str(self.id)+'_Uf' )
        self._bf = initializations.get( 'ones' )( (self._n_out), name=str(self.id)+'_bf' )
        self._Wo = initializations.get( self._init_type )( (n_in, self._n_out), name=str(self.id)+'_Wo' )
        self._Uo = initializations.get( self._init_type )( (self._n_out, self._n_out), name=str(self.id)+'_Uo' )
        self._bo = initializations.get( 'zeros' )( (self._n_out), name=str(self.id)+'_bo' )
        
        # scan
        output = self._scan( input )
        
        # mask
        if self._masking==True:
            mask = get_mask( input )
            output *= mask[:,:,None]
        
        # only return the last value, size: batch_size*n_in
        if self._return_sequence is False:
            output = output[:,-1,:].flatten(2)
            out_shape = (None, self._n_out)
        else:
            out_shape = (None, n_time, self._n_out)
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = out_shape
        self._output = output
        self._params = [ self._Wg, self._Ug, self._bg, self._Wi, self._Ui, self._bi, 
                         self._Wf, self._Uf, self._bf, self._Wo, self._Uo, self._bo ]
        self._reg_value = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    # size(input): batch_size*n_times*n_in
    def _scan( self, input ):
        assert input.ndim==3
        N, n_times, n_in = input.shape
        
        # to use scan, dimshuffle input to shape: n_times*batch_size*n_in
        [ss, hs], update = K.scan( self._step, sequences=input.dimshuffle(1,0,2), outputs_info=[ K.zeros((N, self._n_out)), K.zeros((N, self._n_out)) ] )
        
        # dimshuffle output back to shape: batch_size*n_times*n_in
        output = hs.dimshuffle(1,0,2)
        return output
        
    def _step( self, x, s_, h_ ):
        g = activations.get( self._act )( K.dot( x, self._Wg ) + K.dot( h_, self._Ug ) + self._bg )
        i = K.sigmoid( K.dot( x, self._Wi ) + K.dot( h_, self._Ui ) + self._bi )
        f = K.sigmoid( K.dot( x, self._Wf ) + K.dot( h_, self._Uf ) + self._bf )
        o = K.sigmoid( K.dot( x, self._Wo ) + K.dot( h_, self._Uo ) + self._bo )
        s = g * i + s_ * f
        h = activations.get( self._act )( s ) * o
        return s, h
        
    # get regularization
    def _get_reg( self ):
        if self._reg is None:
            _reg_value = 0.
        else:
            _reg_value = self._reg.reg_value( [ self._Wg, self._Ug, self._bg, self._Wi, self._Ui, self._bi, 
                                                self._Wf, self._Uf, self._bf, self._Wo, self._Uo, self._bo ] )
        return _reg_value
        
'''
Gated Recurrent Unit
Ref. [1] Cho, Kyunghyun, et al. "Learning phrase representations using RNN encoder-decoder for statistical machine translation."
'''
class GRU( RnnBase ):
    def __init__( self, n_out, act, init_type='uniform', reg=None, return_sequence=True, go_backwards=False, masking=False, name=None ):
        super( GRU, self ).__init__( n_out, act, init_type, reg, return_sequence, go_backwards, masking, name )
        
    def __call__( self, in_layers ):
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Rnn can only be one layer!"
        in_layer = in_layers[0]
        
        in_shape = in_layer.out_shape
        input = in_layer.output
        assert len(in_shape)==3, "The dim of input must be 3! Your shape is " + str(in_shape)
        [ batch_size, n_time, n_in ] = in_shape
        
        # reverse data
        if self._go_backwards: input = input[::-1]
        
        # parameters
        self._Wr = initializations.get( self._init_type )( (n_in, self._n_out), name=str(self.id)+'_Wr' )
        self._Ur = initializations.get( self._init_type )( (self._n_out, self._n_out), name=str(self.id)+'_Ur' )
        self._Wz = initializations.get( self._init_type )( (n_in, self._n_out), name=str(self.id)+'_Wz' )
        self._Uz = initializations.get( self._init_type )( (self._n_out, self._n_out), name=str(self.id)+'_Uz' )
        self._W = initializations.get( self._init_type )( (n_in, self._n_out), name=str(self.id)+'_W' )
        self._U = initializations.get( self._init_type )( (self._n_out, self._n_out), name=str(self.id)+'_U' )
        
        # scan
        output = self._scan( input )
        
        # mask
        if self._masking==True:
            mask = get_mask( input )
            output *= mask[:,:,None]
        
        # only return the last value, size: batch_size*n_in
        if self._return_sequence is False:
            output = output[:,-1,:].flatten(2)
            out_shape = (None, self._n_out)
        else:
            out_shape = (None, n_time, self._n_out)
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = out_shape
        self._output = output
        self._params = [ self._Wr, self._Ur, self._Wz, self._Uz, self._W, self._U ]
        self._reg_value = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    # size(input): batch_size*n_times*n_in
    def _scan( self, input ):
        assert input.ndim==3
        N, n_times, n_in = input.shape
        
        # to use scan, dimshuffle input to shape: n_times*batch_size*n_in
        hs, update = K.scan( self._step, sequences=input.dimshuffle(1,0,2), outputs_info=[ K.zeros((N, self._n_out)) ] )
        
        # dimshuffle output back to shape: batch_size*n_times*n_in
        output = hs.dimshuffle(1,0,2)
        return output
        
    def _step( self, x, h_ ):
        r = K.sigmoid( K.dot( x, self._Wr ) + K.dot( h_, self._Ur ) )
        z = K.sigmoid( K.dot( x, self._Wz ) + K.dot( h_, self._Uz ) )
        h_tilde = activations.get( self._act )( K.dot( x, self._W ) + K.dot( r*h_, self._U ) )
        h = z * h_ + ( 1 - z ) * h_tilde
        return h
        
    # get regularization
    def _get_reg( self ):
        if self._reg is None:
            _reg_value = 0.
        else:
            _reg_value = self._reg.reg_value( [ self._Wr, self._Ur, self._Wz, self._Uz, self._W, self._U ] )
        return _reg_value