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

###
# Base class of all variations of RNN
class RnnBase( Layer ):
    __metaclass__ = ABCMeta
    
    def __init__( self, n_out, act, init_type, reg, return_sequence, go_backwards, masking, name ):
        super( RnnBase, self ).__init__( name )
        self._n_out_ = n_out
        self._act_ = act
        self._init_type_ = init_type
        self._reg_ = reg
        self._return_sequence_ = return_sequence
        self._go_backwards_ = go_backwards
        self._masking_ = masking
    
    @abstractmethod
    def _step( self ):
        pass

    @abstractmethod
    def _scan( self ):
        pass
        
'''
Simple Rnn layer. 
(Using random matrix to init H is better than eye matrix)
'''
class SimpleRnn( RnnBase ):
    def __init__( self, n_out, act, init_type='uniform', reg=None,
                  W_init=None, H_init=None, b_init=None, 
                  return_sequence=True, go_backwards=False, masking=False, name=None ):
        
        super( SimpleRnn, self ).__init__( n_out, act, init_type, reg, return_sequence, go_backwards, masking, name )
        self._W_init_ = W_init
        self._H_init_ = H_init
        self._b_init_ = b_init
        
    def __call__( self, in_layers ):
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Rnn can only be one layer!"
        in_layer = in_layers[0]
        
        in_shape = in_layer.out_shape_
        input = in_layer.output_
        assert len(in_shape)==3, "The dim of input must be 3! Your shape is " + str(in_shape)
        [ batch_size, n_time, n_in ] = in_shape
        
        # reverse data
        if self._go_backwards_: input = input[::-1]
        
        # ------ init parameters ------
        # init W
        if self._W_init_ is None:
            self._W_ = initializations.get( self._init_type_ )( (n_in, self._n_out_), name=str(self._name_)+'_W' )
        else:
            self._W_ = K.sh_variable( self._W_init_ )

        # init H
        if self._H_init_ is None:
            self._H_ = initializations.get( self._init_type_ )( (self._n_out_, self._n_out_), name=str(self._name_)+'_H' )
        else:
            self._H_ = K.sh_variable( self._H_init_ )
            
        # init b
        if self._b_init_ is None:
            self._b_ = initializations.get( 'zeros' )( (self._n_out_), name=str(self._name_)+'_b' )
        else:
            self._b_ = K.sh_variable( self._b_init_ )
     
        # ------------------------------
        
        # scan
        output = self._scan( input )
        
        # mask
        if self._masking_==True:
            mask = get_mask( input )
            output *= mask[:,:,None]
        
        # if return_sequence=False only return the last value, size: batch_size*n_in
        if self._return_sequence_ is False:
            output = output[:,-1,:].flatten(2)
            out_shape = (None, self._n_out_)
        else:
            out_shape = (None, n_time, self._n_out_)
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = out_shape
        self._output_ = output
        self._params_ = [ self._W_, self._b_, self._H_ ]
        self._reg_value_ = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                             # check if all attributes are implemented
        return self
    
    # ---------- Public attributes ----------
    
    @property
    def W_( self ): return K.get_value( self._W_ )
        
    @property
    def H_( self ): return K.get_value( self._H_ )
        
    @property
    def b_( self ): return K.get_value( self._b_ )
        
    # ---------- Private methods ----------
    
    # size(input): batch_size*n_times*n_in
    def _scan( self, input ):
        assert input.ndim==3
        batch_size, n_times, n_in = input.shape
         
        # to use scan, dimshuffle input to shape: n_times*batch_size*n_in
        results, update = K.scan( self._step, sequences=input.dimshuffle(1,0,2), outputs_info=[K.zeros((batch_size, self._n_out_))] )
        
        # dimshuffle output back to shape: batch_size*n_times*n_in
        output = results.dimshuffle(1,0,2)
        return output
    
    # size(x): batch_size*n_in, size(h_): batch_size*n_out
    def _step( self, x, h_ ):
        lin_out = K.dot( x, self._W_ ) + K.dot( h_, self._H_ ) + self._b_
        h = activations.get( self._act_ )( lin_out )

        return h
        
    # get regularization
    def _get_reg( self ):
        if self._reg_ is None:
            _reg_value_ = 0.
        else:
            _reg_value_ = self._reg_.reg_value( [ self.W, self.H ] )
        return _reg_value_
        
    # ---------- Public methods ----------
    
    # model's info & params
    @property
    def info_( self ):
        dict = { 'id': self._id_, 
                 'n_out': self._n_out_, 
                 'act': self._act_, 
                 'init_type': self._init_type_, 
                 'reg': self._reg_, 
                 'return_sequence': self._return_sequence_, 
                 'go_backwards': self._go_backwards_, 
                 'masking': self._masking_, 
                 'name': self._name_, 
                 'W': self.W_ , 
                 'H': self.H_, 
                 'b': self.b_, }
        return dict
           
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        layer = cls( n_out=info['n_out'], act=info['act'], init_type=info['init_type'], 
                     reg=info['reg'], W_init=info['W'], H_init=info['H'], b_init=info['b'], 
                     return_sequence=info['return_sequence'], go_backwards=info['go_backwards'], 
                     masking=info['masking'], name=info['name'] )
        return layer
    
    # -------------------------------------
        
'''
[1] Hochreiter, "Long short-term memory." (1997)
[2] 
'''
class LSTM( RnnBase ):
    def __init__( self, n_out, act, init_type='uniform', reg=None, 
                  Wg_init=None, Ug_init=None, bg_init=None, Wi_init=None, Ui_init=None, bi_init=None, 
                  Wf_init=None, Uf_init=None, bf_init=None, Wo_init=None, Uo_init=None, bo_init=None, 
                  return_sequence=True, go_backwards=False, masking=False, name=None ):
        super( LSTM, self ).__init__( n_out, act, init_type, reg, return_sequence, go_backwards, masking, name )
        self._Wg_init_ = Wg_init
        self._Ug_init_ = Ug_init
        self._bg_init_ = bg_init
        self._Wi_init_ = Wi_init
        self._Ui_init_ = Ui_init
        self._bi_init_ = bi_init
        self._Wf_init_ = Wf_init
        self._Uf_init_ = Uf_init
        self._bf_init_ = bf_init
        self._Wo_init_ = Wo_init
        self._Uo_init_ = Uo_init
        self._bo_init_ = bo_init
        
    
            #return K.
        
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
        
        # ------ init parameters ------
        if self._Wg_init_ is None:
            self._Wg_ = initializations.get( self._init_type )( (n_in, self._n_out), name=str(self.id)+'_Wg' )
        else:
            self._Wg_ = self._Wg_init_
            
        if self._Ug_init_ is None:
            self._Ug_ = initializations.get( self._init_type )( (self._n_out, self._n_out), name=str(self.id)+'_Ug' )
        else:
            self._Ug_ = self._Ug_init_
            
        if self._bg_init_ is None:
            self._bg_ = initializations.get( 'zeros' )( (self._n_out), name=str(self.id)+'_bg' )
        else:
            self._bg_ = self._bg_init_
            
        if self._Wi_init_ is None:
            self._Wi_ = initializations.get( self._init_type )( (n_in, self._n_out), name=str(self.id)+'_Wi' )
        else:
            self._Wi_ = self._Wi_init_
            
        if self._Wi_init_ is None:
            self._Ui_ = initializations.get( self._init_type )( (self._n_out, self._n_out), name=str(self.id)+'_Ui' )
        else:
            self._Ui_ = self._Ui_init_
            
        if self._bi_init_ is None:
            self._bi_ = initializations.get( 'zeros' )( (self._n_out), name=str(self.id)+'_bi' )
        else:
            self._bi_ = self._bi_init_
            
            
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
        self._check_attributes()                             # check if all attributes are implemented
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
        self._check_attributes()                             # check if all attributes are implemented
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