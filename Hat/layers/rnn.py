'''
SUMMARY:  Rnn, Lstm
AUTHOR:   Qiuqiang Kong
Created:  2016.05.17
Modified: 2016.05.21 Modify bug in LSTM
          2016.08.03 Add regularization, serialization to all rnn
--------------------------------------
'''
from core import Layer
from ..globals import new_id
from .. import backend as K
from .. import initializations
from .. import activations
from .. import regularizations
from ..supports import to_list, get_mask
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

###
''' 
Base class of all variations of RNN
'''
class RnnBase( Layer ):
    __metaclass__ = ABCMeta
    
    def __init__( self, n_out, act, return_sequence, go_backwards, masking, name ):
        super( RnnBase, self ).__init__( name )
        self._n_out_ = n_out
        self._act_ = act
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
    def __init__( self, n_out, act, W_init_type='uniform', H_init_type='uniform', 
                  W_init=None, H_init=None, b_init=None, W_reg=None, H_reg=None, b_reg=None, 
                  return_sequence=True, go_backwards=False, masking=False, name=None ):
        
        super( SimpleRnn, self ).__init__( n_out, act, return_sequence, go_backwards, masking, name )
        self._W_init_type_ = W_init_type
        self._H_init_type_ = H_init_type
        self._W_init_ = W_init
        self._H_init_ = H_init
        self._b_init_ = b_init
        self._W_reg_ = W_reg
        self._H_reg_ = H_reg
        self._b_reg_ = b_reg
        
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
        
        # init parameters
        self._W_ = self._init_params( self._W_init_, self._W_init_type_, (n_in, self._n_out_), name=str(self._name_)+'_W' )
        self._H_ = self._init_params( self._H_init_, self._H_init_type_, (self._n_out_, self._n_out_), name=str(self._name_)+'_H' )
        self._b_ = self._init_params( self._b_init_, 'zeros', (self._n_out_,), name=str(self._name_)+'_b' )
        
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
        self._params_ = [ self._W_, self._H_, self._b_ ]
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
        
    # model's info & params
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'n_out': self._n_out_, 
                 'act': self._act_, 
                 'W_init_type': self._W_init_type_, 
                 'H_init_type': self._H_init_type_, 
                 'W': self.W_ , 
                 'H': self.H_, 
                 'b': self.b_, 
                 'W_reg_info': regularizations.get_info( self._W_reg_ ), 
                 'H_reg_info': regularizations.get_info( self._H_reg_ ), 
                 'b_reg_info': regularizations.get_info( self._b_reg_ ), 
                 'return_sequence': self._return_sequence_, 
                 'go_backwards': self._go_backwards_, 
                 'masking': self._masking_, 
                 'name': self._name_, 
                  }
        return dict
        
    # ---------- Public methods ----------
    
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        W_reg = regularizations.get_obj( info['W_reg_info'] )
        H_reg = regularizations.get_obj( info['H_reg_info'] )
        b_reg = regularizations.get_obj( info['b_reg_info'] )

        layer = cls( n_out=info['n_out'], act=info['act'], 
                     W_init_type=info['W_init_type'], H_init_type=info['H_init_type'], 
                     W_init=info['W'], H_init=info['H'], b_init=info['b'], 
                     W_reg=W_reg, H_reg=H_reg, b_reg=b_reg, 
                     return_sequence=info['return_sequence'], go_backwards=info['go_backwards'], 
                     masking=info['masking'], name=info['name'] )
                     
        return layer
        
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
        reg_value = 0. 
        
        if self._W_reg_ is not None:
            reg_value += self._W_reg_.get_reg( [self._W_] )
        
        if self._H_reg_ is not None:
            reg_value += self._H_reg_.get_reg( [self._H_] )
            
        if self._b_reg_ is not None:
            reg_value += self._b_reg_.get_reg( [self._b_] )
            
        return reg_value
        
    # -------------------------------------
        
'''
[1] Hochreiter, "Long short-term memory." (1997)
[2] Lipton, "A critical review of recurrent neural networks for sequence learning." (2015).
[3] Zaremba, Wojciech. "An empirical exploration of recurrent network architectures." (2015).
'''
class LSTM( RnnBase ):
    def __init__( self, n_out, act, gate_act='sigmoid', W_init_type='uniform', H_init_type='uniform', bf_bias=1., 
                  Wg_init=None, Hg_init=None, bg_init=None, Wi_init=None, Hi_init=None, bi_init=None, 
                  Wf_init=None, Hf_init=None, bf_init=None, Wo_init=None, Ho_init=None, bo_init=None, 
                  W_reg=None, H_reg=None, return_sequence=True, go_backwards=False, masking=False, name=None ):
                      
        super( LSTM, self ).__init__( n_out, act, return_sequence, go_backwards, masking, name )
        self._gate_act_ = gate_act
        self._W_init_type_ = W_init_type
        self._H_init_type_ = H_init_type
        self._bf_bias_ = bf_bias
        self._Wg_init_ = Wg_init
        self._Hg_init_ = Hg_init
        self._bg_init_ = bg_init
        self._Wi_init_ = Wi_init
        self._Hi_init_ = Hi_init
        self._bi_init_ = bi_init
        self._Wf_init_ = Wf_init
        self._Hf_init_ = Hf_init
        self._bf_init_ = bf_init
        self._Wo_init_ = Wo_init
        self._Ho_init_ = Ho_init
        self._bo_init_ = bo_init
        self._W_reg_ = W_reg
        self._H_reg_ = H_reg
        
    def __call__( self, in_layers ):
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Rnn can only be one layer!"
        in_layer = in_layers[0]
        
        # prepare data
        in_shape = in_layer.out_shape_
        input = in_layer.output_
        assert len(in_shape)==3, "The dim of input must be 3! Your shape is " + str(in_shape)
        [ batch_size, n_time, n_in ] = in_shape
        
        # reverse data
        if self._go_backwards_: input = input[::-1]
        
        # ------ init parameters ------
        self._Wg_ = self._init_params( self._Wg_init_, self._W_init_type_, (n_in, self._n_out_), name=str(self.id_)+'_Wg' ) 
        self._Hg_ = self._init_params( self._Hg_init_, self._H_init_type_, (self._n_out_, self._n_out_), name=str(self.id_)+'_Hg' )
        self._bg_ = self._init_params( self._bg_init_, 'zeros', (self._n_out_,), name=str(self.id_)+'_bg' )
        self._Wi_ = self._init_params( self._Wi_init_, self._W_init_type_, (n_in, self._n_out_), name=str(self.id_)+'_Wi' )
        self._Hi_ = self._init_params( self._Hi_init_, self._H_init_type_, (self._n_out_, self._n_out_), name=str(self.id_)+'_Hi' )
        self._bi_ = self._init_params( self._bi_init_, 'zeros', (self._n_out_,), name=str(self.id_)+'_bi' )
        self._Wf_ = self._init_params( self._Wf_init_, self._W_init_type_, (n_in, self._n_out_), name=str(self.id_)+'_Wf' )
        self._Hf_ = self._init_params( self._Hf_init_, self._H_init_type_, (self._n_out_, self._n_out_), name=str(self.id_)+'_Hf' )
        
        if self._bf_init_ is None: self._bf_ = K.shared( self._bf_bias_*np.ones(self._n_out_), name=str(self.id_)+'_bf' )
        else: self._bf_ = K.shared( self._bf_init_, name=str(self.id_)+'_bf' )
        
        self._Wo_ = self._init_params( self._Wo_init_, self._W_init_type_, (n_in, self._n_out_), name=str(self.id_)+'_Wo' )
        self._Ho_ = self._init_params( self._Ho_init_, self._H_init_type_, (self._n_out_, self._n_out_), name=str(self.id_)+'_Ho' )
        self._bo_ = self._init_params( self._bo_init_, 'zeros', (self._n_out_,), name=str(self.id_)+'_bo' )
 
        # scan
        output = self._scan( input )
        
        # mask
        if self._masking_==True:
            mask = get_mask( input )
            output *= mask[:,:,None]
        
        # only return the last value, size: batch_size*n_in
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
        self._params_ = [ self._Wg_, self._Hg_, self._bg_, self._Wi_, self._Hi_, self._bi_, 
                         self._Wf_, self._Hf_, self._bf_, self._Wo_, self._Ho_, self._bo_ ]
        self._reg_value_ = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                             # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
    
    @property
    def Wg_( self ): return K.get_value( self._Wg_ )
    
    @property
    def Hg_( self ): return K.get_value( self._Hg_ )
    
    @property
    def bg_( self ): return K.get_value( self._bg_ )
    
    @property
    def Wi_( self ): return K.get_value( self._Wi_ )
    
    @property
    def Hi_( self ): return K.get_value( self._Hi_ )
    
    @property
    def bi_( self ): return K.get_value( self._bi_ )
    
    @property
    def Wf_( self ): return K.get_value( self._Wf_ )
    
    @property
    def Hf_( self ): return K.get_value( self._Hf_ )
    
    @property
    def bf_( self ): return K.get_value( self._bf_ )
    
    @property
    def Wo_( self ): return K.get_value( self._Wo_ )
    
    @property
    def Ho_( self ): return K.get_value( self._Ho_ )
    
    @property
    def bo_( self ): return K.get_value( self._bo_ )
        
    # model's info & params
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'n_out': self._n_out_, 
                 'act': self._act_, 
                 'gate_act': self._gate_act_, 
                 'W_init_type': self._W_init_type_, 
                 'H_init_type': self._H_init_type_, 
                 'bf_bias': self._bf_bias_, 
                 'Wg': self.Wg_, 'Hg': self.Hg_, 'bg': self.bg_, 
                 'Wi': self.Wi_, 'Hi': self.Hi_, 'bi': self.bi_, 
                 'Wf': self.Wf_, 'Hf': self.Hf_, 'bf': self.bf_, 
                 'Wo': self.Wo_, 'Ho': self.Ho_, 'bo': self.bo_, 
                 'W_reg_info': regularizations.get_info( self._W_reg_ ), 
                 'H_reg_info': regularizations.get_info( self._H_reg_ ), 
                 'return_sequence': self._return_sequence_, 
                 'go_backwards': self._go_backwards_, 
                 'masking': self._masking_, 
                 'name': self._name_, 
                  }
        return dict
        
    # ---------- Public methods ----------
    
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        W_reg = regularizations.get_obj( info['W_reg_info'] )
        H_reg = regularizations.get_obj( info['H_reg_info'] )

        layer = cls( n_out=info['n_out'], act=info['act'], gate_act=info['gate_act'],
                     W_init_type=info['W_init_type'], H_init_type=info['H_init_type'], bf_bias=info['bf_bias'], 
                     Wg_init=info['Wg'], Hg_init=info['Hg'], bg_init=info['bg'], 
                     Wi_init=info['Wi'], Hi_init=info['Hi'], bi_init=info['bi'], 
                     Wf_init=info['Wf'], Hf_init=info['Hf'], bf_init=info['bf'], 
                     Wo_init=info['Wo'], Ho_init=info['Ho'], bo_init=info['bo'], 
                     W_reg=W_reg, H_reg=H_reg, 
                     return_sequence=info['return_sequence'], go_backwards=info['go_backwards'], 
                     masking=info['masking'], name=info['name'] )
                     
        return layer
        
    # ---------- Private methods ----------
        
    # size(input): batch_size*n_times*n_in
    def _scan( self, input ):
        assert input.ndim==3
        N, n_times, n_in = input.shape
        
        # to use scan, dimshuffle input to shape: n_times*batch_size*n_in
        [ss, hs], update = K.scan( self._step, sequences=input.dimshuffle(1,0,2), outputs_info=[ K.zeros((N, self._n_out_)), K.zeros((N, self._n_out_)) ] )
        
        # dimshuffle output back to shape: batch_size*n_times*n_in
        output = hs.dimshuffle(1,0,2)
        return output
        
    def _step( self, x, s_, h_ ):
        g = activations.get( self._act_ )( K.dot( x, self._Wg_ ) + K.dot( h_, self._Hg_ ) + self._bg_ )
        i = activations.get( self._gate_act_ )( K.dot( x, self._Wi_ ) + K.dot( h_, self._Hi_ ) + self._bi_ )
        f = activations.get( self._gate_act_ )( K.dot( x, self._Wf_ ) + K.dot( h_, self._Hf_ ) + self._bf_ )
        o = activations.get( self._gate_act_ )( K.dot( x, self._Wo_ ) + K.dot( h_, self._Ho_ ) + self._bo_ )
        s = g * i + s_ * f
        h = s * o
        return s, h
        
    # get regularization
    def _get_reg( self ):
        reg_value = 0. 
        
        if self._W_reg_ is not None:
            reg_value += self._W_reg_.get_reg( [self._Wg_, self._Wi_, self._Wf_, self._Wo_] )
        
        if self._H_reg_ is not None:
            reg_value += self._H_reg_.get_reg( [self._Hg_, self._Hi_, self._Hf_, self._Ho_] )
            
        return reg_value
        
    # -------------------------------------
        
        
        
'''
Gated Recurrent Unit
Ref. [1] Cho, Kyunghyun, et al. "Learning phrase representations using RNN encoder-decoder for statistical machine translation."
'''
class GRU( RnnBase ):
    def __init__( self, n_out, act, gate_act='sigmoid', W_init_type='uniform', U_init_type='uniform', 
                  Wr_init=None, Ur_init=None, br_init=None, Wz_init=None, Uz_init=None, bz_init=None, 
                  W_init=None, U_init=None, b_init=None, W_reg=None, U_reg=None, 
                  return_sequence=True, go_backwards=False, masking=False, name=None ):
        super( GRU, self ).__init__( n_out, act, return_sequence, go_backwards, masking, name )
        self._gate_act_ = gate_act
        self._W_init_type_ = W_init_type
        self._U_init_type_ = U_init_type
        self._Wr_init_ = Wr_init
        self._Ur_init_ = Ur_init
        self._br_init_ = br_init
        self._Wz_init_ = Wz_init
        self._Uz_init_ = Uz_init
        self._bz_init_ = bz_init
        self._W_init_ = W_init
        self._U_init_ = U_init
        self._b_init_ = b_init
        self._W_reg_ = W_reg
        self._U_reg_ = U_reg
        
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
        
        # parameters
        self._Wr_ = self._init_params( self._Wr_init_, self._W_init_type_, (n_in, self._n_out_), name=str(self.id_)+'_Wr' )
        self._Ur_ = self._init_params( self._Ur_init_, self._U_init_type_, (self._n_out_, self._n_out_), name=str(self.id_)+'_Ur' )
        self._br_ = self._init_params( self._br_init_, 'zeros', (self._n_out_,), name=str(self.id_)+'_br' )
        self._Wz_ = self._init_params( self._Wz_init_, self._W_init_type_, (n_in, self._n_out_), name=str(self.id_)+'_Wz' )
        self._Uz_ = self._init_params( self._Uz_init_, self._U_init_type_, (self._n_out_, self._n_out_), name=str(self.id_)+'_Uz' )
        self._bz_ = self._init_params( self._bz_init_, 'zeros', (self._n_out_,), name=str(self.id_)+'_bz' )
        self._W_ = self._init_params( self._W_init_, self._W_init_type_, (n_in, self._n_out_), name=str(self.id_)+'_W' )
        self._U_ = self._init_params( self._U_init_, self._U_init_type_, (self._n_out_, self._n_out_), name=str(self.id_)+'_U' )
        self._b_ = self._init_params( self._b_init_, 'zeros', (self._n_out_,), name=str(self.id_)+'_b' )
        
        # scan
        output = self._scan( input )
        
        # mask
        if self._masking_==True:
            mask = get_mask( input )
            output *= mask[:,:,None]
        
        # only return the last value, size: batch_size*n_in
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
        self._params_ = [ self._Wr_, self._Ur_, self._br_, self._Wz_, self._Uz_, self._bz_, 
                         self._W_, self._U_, self._b_ ]
        self._reg_value_ = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                             # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
    
    @property
    def Wr_( self ): return K.get_value( self._Wr_ )
    
    @property
    def Ur_( self ): return K.get_value( self._Ur_ )
    
    @property
    def br_( self ): return K.get_value( self._br_ )
    
    @property
    def Wz_( self ): return K.get_value( self._Wz_ )
    
    @property
    def Uz_( self ): return K.get_value( self._Uz_ )
    
    @property
    def bz_( self ): return K.get_value( self._bz_ )
    
    @property
    def W_( self ): return K.get_value( self._W_ )
    
    @property
    def U_( self ): return K.get_value( self._U_ )
    
    @property
    def b_( self ): return K.get_value( self._b_ )
    
    # model's info & params
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'n_out': self._n_out_, 
                 'act': self._act_, 
                 'gate_act': self._gate_act_, 
                 'W_init_type': self._W_init_type_, 
                 'U_init_type': self._U_init_type_, 
                 'Wr': self.Wr_, 'Ur': self.Ur_, 'br': self.br_, 
                 'Wz': self.Wz_, 'Uz': self.Uz_, 'bz': self.bz_, 
                 'W': self.W_, 'U': self.U_, 'b': self.b_, 
                 'W_reg_info': regularizations.get_info( self._W_reg_ ), 
                 'U_reg_info': regularizations.get_info( self._U_reg_ ), 
                 'return_sequence': self._return_sequence_, 
                 'go_backwards': self._go_backwards_, 
                 'masking': self._masking_, 
                 'name': self._name_, 
                  }
        return dict
        
    # ---------- Public methods ----------
    
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        W_reg = regularizations.get_obj( info['W_reg_info'] )
        U_reg = regularizations.get_obj( info['U_reg_info'] )

        layer = cls( n_out=info['n_out'], act=info['act'], gate_act=info['gate_act'],
                     W_init_type=info['W_init_type'], U_init_type=info['U_init_type'], 
                     Wr_init=info['Wr'], Ur_init=info['Ur'], br_init=info['br'], 
                     Wz_init=info['Wz'], Uz_init=info['Uz'], bz_init=info['bz'], 
                     W_init=info['W'], U_init=info['U'], b_init=info['b'], 
                     W_reg=W_reg, U_reg=U_reg, 
                     return_sequence=info['return_sequence'], go_backwards=info['go_backwards'], 
                     masking=info['masking'], name=info['name'] )
                     
        return layer
        
    # ---------- Private methods ----------
        
    # size(input): batch_size*n_times*n_in
    def _scan( self, input ):
        assert input.ndim==3
        N, n_times, n_in = input.shape
        
        # to use scan, dimshuffle input to shape: n_times*batch_size*n_in
        hs, update = K.scan( self._step, sequences=input.dimshuffle(1,0,2), outputs_info=[ K.zeros((N, self._n_out_)) ] )
        
        # dimshuffle output back to shape: batch_size*n_times*n_in
        output = hs.dimshuffle(1,0,2)
        return output
        
    def _step( self, x, h_ ):
        r = K.sigmoid( K.dot( x, self._Wr_ ) + K.dot( h_, self._Ur_ ) + self._br_ )
        z = K.sigmoid( K.dot( x, self._Wz_ ) + K.dot( h_, self._Uz_ ) + self._bz_ )
        h_tilde = activations.get( self._act_ )( K.dot( x, self._W_ ) + K.dot( r*h_, self._U_ ) + self._b_ )
        h = z * h_ + ( 1 - z ) * h_tilde
        return h
        
    # get regularization
    def _get_reg( self ):
        reg_value = 0. 
        
        if self._W_reg_ is not None:
            reg_value += self._W_reg_.get_reg( [self._Wr_, self._Wz_, self._W_] )
        
        if self._U_reg_ is not None:
            reg_value += self._H_reg_.get_reg( [self._Ur_, self._Uz_, self._U_] )
            
        return reg_value
        
    # -------------------------------------