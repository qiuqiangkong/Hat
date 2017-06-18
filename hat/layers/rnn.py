'''
SUMMARY:  Rnn, Lstm
AUTHOR:   Qiuqiang Kong
Created:  2016.05.17
Modified: 2016.05.21 Modify bug in LSTM
          2016.08.03 Add regularization, serialization to all rnn
          2016.08.26 Modify SimpleRnn to high dimension version
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


class RnnBase(Layer):
    __metaclass__ = ABCMeta

    def __init__(self, kernel_init_type='glorot_uniform', 
                  recurrent_init_type='orthogonal', 
                  return_sequences=False, 
                  go_backwards=False, 
                  stateful=False, 
                  masking=False, 
                  **kwargs):
        super(RnnBase, self).__init__(**kwargs)
        self._check_kwargs(kwargs)
        self._kernel_init_type_ = kernel_init_type
        self._recurrent_init_type_ = recurrent_init_type
        self._return_sequences_ = return_sequences
        self._go_backwards_ = go_backwards
        self._stateful_ = stateful
        self._masking_ = masking
      
    # ---------- Public attributes ----------
    
    @property
    def info_(self):
        info = {'kernel_init_type': self._kernel_init_type_, 
                'recurrent_init_type': self._recurrent_init_type_, 
                'return_sequences': self._return_sequences_, 
                'go_backwards': self._go_backwards_, 
                'stateful': self._stateful_, 
                'masking': self._masking_
               }
        return info

    # ---------- Private methods ----------
        
    # @abstractmethod
    def _step(self):
        raise NotImplementedError
        
    def _get_out_shape(self, in_shape):
        if self._return_sequences_:
            return in_shape[0:2] + (self._n_out_,)  # shape: (n_batch, n_time, n_out)
        else:
            return in_shape[0:1] + (self._n_out_,)  # shape: (n_batch, n_out)
     
    # ------------------------------------
     
        
class SimpleRNN(RnnBase):
    # TODO stateful, info
    def __init__(self, n_out, act, W_init=None, H_init=None, b_init=None, 
                 trainable_params=['W','H','b'], 
                 **kwargs):
        self._legal_params_ = ['W', 'H', 'b']
        super(SimpleRNN, self).__init__(**kwargs)
        self._n_out_ = n_out
        self._act_ = act
        self._W_init_ = W_init
        self._H_init_ = H_init
        self._b_init_ = b_init
        self._trainable_params_ = trainable_params
        
    def __call__(self, in_layers):
        in_layers = to_list(in_layers)
        assert len(in_layers)==1, "The input of Rnn can only be one layer!"
        in_shape = in_layers[0].out_shape_
        assert len(in_shape)==3, "The input dimension should be 3 to Rnn layer!"
        [batch_size, n_time, n_in] = in_shape
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = self._get_out_shape(in_shape)
        
        # shared params
        if not hasattr(self, '_W_'):
            self._W_ = self._init_params(self._W_init_, self._kernel_init_type_, 
                                        shape=(n_in, self._n_out_), 
                                        name=str(self._name_)+'_W')
        if not hasattr(self, '_H_'):
            self._H_ = self._init_params(self._H_init_, self._recurrent_init_type_, 
                                        shape=(self._n_out_, self._n_out_), 
                                        name=str(self._name_)+'_H')
        if not hasattr(self, '_b_'):
            self._b_ = self._init_params(self._b_init_, 'zeros', 
                                        shape=(self._n_out_,), 
                                        name=str(self._name_)+'_b')
        self._init_h_ = None  # will be allocated in compile()
                                                                 
        # set params & update reg_value
        self.set_trainable_params_and_update_reg(self._trainable_params_)
        
        # below are compulsory parts
        self._add_self_to_prevs_layer(in_layers)     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                       # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
        
    @property
    def info_(self):
        rnn_base_info = super(SimpleRNN, self).info_
        info = {'class_name': self.__class__.__name__,
                'n_out': self._n_out_, 
                'act': self._act_, 
                'W_init': self.W_, 
                'H_init': self.H_, 
                'b_init': self.b_, 
                'trainable_params': self._trainable_params_, 
                'kwargs': self._base_kwargs_
                }
        info.update(rnn_base_info)
        return info
        
    @property    
    def W_(self):
        return K.get_value(self._W_)
        
    @property
    def b_(self):
        return K.get_value(self._b_)
    
    @property    
    def H_(self):
        return K.get_value(self._H_)
        
    # ---------- Public methods ----------
 
    def compile(self):
        self._inputs_ =[layer.output_ for layer in self._prevs_]
        input = self._inputs_[0]                                         
        self._output_ = self._get_output(input)
  
    @classmethod
    def load_from_info(cls, info):
        layer = cls(n_out=info['n_out'], 
                    act=info['act'], 
                    W_init=info['W_init'], 
                    H_init=info['H_init'], 
                    b_init=info['b_init'], 
                    trainable_params=info['trainable_params'], 
                    kernel_init_type=info['kernel_init_type'], 
                    recurrent_init_type=info['recurrent_init_type'], 
                    return_sequences=info['return_sequences'], 
                    go_backwards=info['go_backwards'], 
                    stateful=info['stateful'], 
                    masking=info['masking'], 
                    **info['kwargs'])
        return layer
        
    # ---------- Private methods ----------
    
    def _step(self, x, h_):
        h = K.dot(x, self._W_) + self._b_
        output = h + K.dot(h_, self._H_)
        out = activations.get(self._act_)(output)
        return out
    
    def _get_reg(self):
        reg_value = 0.
        return reg_value
        
    def _get_output(self, input):
        self._init_h_ = K.zeros((input.shape[0], self._n_out_))     # (n_samples, n_out)
        last_output, output, state = K.rnn(self._step, input, self._init_h_, self._go_backwards_)
        if self._return_sequences_:
            return output
        else:
            return last_output
    
    
class LSTM(RnnBase):
    def __init__(self, n_out, act, gate_act='hard_sigmoid', W_init=None, 
                 U_init=None, b_init=None, forget_bias_init=1., 
                 trainable_params=['W','U','b'], **kwargs):
        self._legal_params_ = ['W', 'U', 'b']
        super(LSTM, self).__init__(**kwargs)
        self._n_out_ = n_out
        self._act_ = act
        self._gate_act_ = gate_act
        self._W_init_ = W_init
        self._U_init_ = U_init
        self._b_init_ = b_init
        self._forget_bias_init_ = forget_bias_init
        self._trainable_params_ = trainable_params
        
    def __call__(self, in_layers):
        in_layers = to_list(in_layers)
        assert len(in_layers)==1, "The input of Rnn can only be one layer!"
        in_shape = in_layers[0].out_shape_
        assert len(in_shape)==3, "The input dimension should be 3 to Rnn layer!"
        [batch_size, n_time, n_in] = in_shape
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = self._get_out_shape(in_shape)
        
        # shared params
        if not hasattr(self, '_W_'):
            self._W_ = self._init_params(self._W_init_, self._kernel_init_type_, 
                                        shape=(n_in, self._n_out_ * 4), 
                                        name=str(self._name_)+'_W')
        if not hasattr(self, '_U_'):
            self._U_ = self._init_params(self._U_init_, self._recurrent_init_type_, 
                                        shape=(self._n_out_, self._n_out_ * 4), 
                                        name=str(self._name_)+'_U')
        if not hasattr(self, '_b_'):
            if self._b_init_ is None:
                np_b = np.concatenate((np.zeros(self._n_out_), 
                                    np.ones(self._n_out_) * self._forget_bias_init_, 
                                    np.zeros(self._n_out_), 
                                    np.zeros(self._n_out_)), 
                                    axis=0)
                self._b_ = K.shared(np_b, name=str(self._name_)+'_b')
                
            else:
                self._b_ = K.shared(self._b_init_, name=str(self._name_)+'_b')
                                              
        # set params & update reg_value
        self.set_trainable_params_and_update_reg(self._trainable_params_)
        
        # below are compulsory parts
        self._add_self_to_prevs_layer(in_layers)     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                       # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
        
    @property
    def info_(self):
        rnn_base_info = super(LSTM, self).info_
        info = {'class_name': self.__class__.__name__,
                'n_out': self._n_out_, 
                'act': self._act_, 
                'gate_act': self._gate_act_, 
                'W_init': self.W_, 
                'U_init': self.U_, 
                'b_init': self.b_, 
                'forget_bias_init': self._forget_bias_init_, 
                'trainable_params': self._trainable_params_, 
                'kwargs': self._base_kwargs_
                }
        info.update(rnn_base_info)
        return info
        
    @property    
    def W_(self):
        return K.get_value(self._W_)
        
    @property
    def b_(self):
        return K.get_value(self._b_)
    
    @property    
    def U_(self):
        return K.get_value(self._U_)
        
    # ---------- Public methods ----------
 
    def compile(self):
        self._inputs_ =[layer.output_ for layer in self._prevs_]
        input = self._inputs_[0]                                         
        self._output_ = self._get_output(input)
  
    @classmethod
    def load_from_info(cls, info):
        layer = cls(n_out=info['n_out'], 
                    act=info['act'],
                    gate_act=info['gate_act'], 
                    W_init=info['W_init'], 
                    U_init=info['U_init'], 
                    b_init=info['b_init'], 
                    forget_bias_init=info['forget_bias_init'], 
                    trainable_params=info['trainable_params'], 
                    kernel_init_type=info['kernel_init_type'], 
                    recurrent_init_type=info['recurrent_init_type'], 
                    return_sequences=info['return_sequences'], 
                    go_backwards=info['go_backwards'], 
                    stateful=info['stateful'], 
                    masking=info['masking'], 
                    **info['kwargs'])
        return layer
        
    def _step(self, x, h_, s_):
        z = K.dot(x, self._W_) + K.dot(h_, self._U_) + self._b_
        zi = z[:, : self._n_out_]
        zf = z[:, self._n_out_ : self._n_out_ * 2]
        zg = z[:, self._n_out_ * 2 : self._n_out_ * 3]
        zo = z[:, self._n_out_ * 3 :]
        g = activations.get(self._act_)(zg)
        i = activations.get(self._gate_act_)(zi)
        o = activations.get(self._gate_act_)(zo)
        f = activations.get(self._gate_act_)(zf)
        s = g * i + s_ * f
        h = o * activations.get(self._act_)(s)
        return h, s
    
    def _get_reg(self):
        reg_value = 0.
        return reg_value
        
    def _get_output(self, input):
        init_s = K.zeros((input.shape[0], self._n_out_))     # (n_samples, n_out)
        init_h = K.zeros((input.shape[0], self._n_out_))     # (n_samples, n_out)
        last_outputs, outputs, states = K.rnn(self._step, input, [init_h, init_s], self._go_backwards_)
        if self._return_sequences_:
            (h, s) = outputs
            return h
        else:
            (last_h, last_s) = last_outputs
            return last_h
            
            
class GRU(RnnBase):
    # TODO stateful, info
    def __init__(self, n_out, act, gate_act='hard_sigmoid', W_init=None, U_init=None, b_init=None, 
                 trainable_params=['W','U','b'], 
                 **kwargs):
        self._legal_params_ = ['W', 'U', 'b']
        super(GRU, self).__init__(**kwargs)
        self._n_out_ = n_out
        self._act_ = act
        self._gate_act_ = gate_act
        self._W_init_ = W_init
        self._U_init_ = U_init
        self._b_init_ = b_init
        self._trainable_params_ = trainable_params
        
    def __call__(self, in_layers):
        in_layers = to_list(in_layers)
        assert len(in_layers)==1, "The input of Rnn can only be one layer!"
        in_shape = in_layers[0].out_shape_
        assert len(in_shape)==3, "The input dimension should be 3 to Rnn layer!"
        [batch_size, n_time, n_in] = in_shape
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = self._get_out_shape(in_shape)
        
        # shared params
        if not hasattr(self, '_W_'):
            self._W_ = self._init_params(self._W_init_, self._kernel_init_type_, 
                                        shape=(n_in, self._n_out_ * 3), 
                                        name=str(self._name_)+'_W')
        if not hasattr(self, '_U_'):
            self._U_ = self._init_params(self._U_init_, self._recurrent_init_type_, 
                                        shape=(self._n_out_, self._n_out_ * 3), 
                                        name=str(self._name_)+'_U')
        if not hasattr(self, '_b_'):
            self._b_ = self._init_params(self._b_init_, 'zeros', 
                                        shape=(self._n_out_ * 3,), 
                                        name=str(self._name_)+'_b')
        self._init_h_ = None  # will be allocated in compile()
        
        self._Wz_ = self._W_[:, 0:self._n_out_]
        self._Wr_ = self._W_[:, self._n_out_:self._n_out_*2]
        self._Wh_ = self._W_[:, self._n_out_*2:]
        self._Uz_ = self._U_[:, 0:self._n_out_]
        self._Ur_ = self._U_[:, self._n_out_:self._n_out_*2]
        self._Uh_ = self._U_[:, self._n_out_*2:]
        self._bz_ = self._b_[0:self._n_out_]
        self._br_ = self._b_[self._n_out_:self._n_out_*2]
        self._bh_ = self._b_[self._n_out_*2:]
                                                                 
        # set params & update reg_value
        self.set_trainable_params_and_update_reg(self._trainable_params_)
        
        # below are compulsory parts
        self._add_self_to_prevs_layer(in_layers)     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                       # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
        
    @property
    def info_(self):
        rnn_base_info = super(GRU, self).info_
        info = {'class_name': self.__class__.__name__,
                'n_out': self._n_out_, 
                'act': self._act_, 
                'gate_act': self._gate_act_, 
                'W_init': self.W_, 
                'U_init': self.U_, 
                'b_init': self.b_, 
                'trainable_params': self._trainable_params_, 
                'kwargs': self._base_kwargs_
                }
        info.update(rnn_base_info)
        return info
        
    @property    
    def W_(self):
        return K.get_value(self._W_)
        
    @property
    def b_(self):
        return K.get_value(self._b_)
    
    @property    
    def U_(self):
        return K.get_value(self._U_)
        
    # ---------- Public methods ----------
 
    def compile(self):
        self._inputs_ =[layer.output_ for layer in self._prevs_]
        input = self._inputs_[0]                                         
        self._output_ = self._get_output(input)
  
    @classmethod
    def load_from_info(cls, info):
        layer = cls(n_out=info['n_out'], 
                    act=info['act'],
                    gate_act=info['gate_act'], 
                    W_init=info['W_init'], 
                    U_init=info['U_init'], 
                    b_init=info['b_init'], 
                    trainable_params=info['trainable_params'], 
                    kernel_init_type=info['kernel_init_type'], 
                    recurrent_init_type=info['recurrent_init_type'], 
                    return_sequences=info['return_sequences'], 
                    go_backwards=info['go_backwards'], 
                    stateful=info['stateful'], 
                    masking=info['masking'], 
                    **info['kwargs'])
        return layer
        
    # ---------- Private methods ----------
    
    def _step(self, x, h_):
        Wzrh_dot_x = K.dot(x, self._W_) + self._b_
        Wz_dot_x = Wzrh_dot_x[:, 0:self._n_out_]
        Wr_dot_x = Wzrh_dot_x[:, self._n_out_:self._n_out_*2]
        Wh_dot_x = Wzrh_dot_x[:, self._n_out_*2:]
        Uzr_dot_prevh = K.dot(h_, self._U_[:, 0:self._n_out_*2])
        Uz_dot_prevh = Uzr_dot_prevh[:, 0:self._n_out_]
        Ur_dot_prevh = Uzr_dot_prevh[:, self._n_out_:self._n_out_*2]
        z = activations.get(self._gate_act_)(Wz_dot_x + Uz_dot_prevh)
        r = activations.get(self._gate_act_)(Wr_dot_x + Ur_dot_prevh)
        Uh = self._U_[:, self._n_out_*2:]
        tild_h = activations.get(self._act_)(K.dot(r*h_, Uh) + Wh_dot_x)
        out = z * h_ + (1 - z) * tild_h     # same as keras, different from original paper. 
        return out
    
    def _get_reg(self):
        reg_value = 0.
        return reg_value
        
    def _get_output(self, input):
        self._init_h_ = K.zeros((input.shape[0], self._n_out_))     # (n_samples, n_out)
        last_output, output, state = K.rnn(self._step, input, self._init_h_, self._go_backwards_)
        if self._return_sequences_:
            return output
        else:
            return last_output