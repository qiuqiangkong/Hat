'''
SUMMARY:  Embedding layers. Eg. Map 20000 vocabulary to 128 vector space
AUTHOR:   Qiuqiang Kong
Created:  2016.05.28
Modified: -
--------------------------------------
'''
from core import Layer
from ..globals import new_id
from .. import backend as K
from .. import initializations
from .. import regularizations
from .. import activations
from ..supports import to_list
import numpy as np

class Embedding(Layer):
    def __init__(self, n_vocab, n_out, init_type='uniform', W_init=None, W_reg=None, 
                 trainable_params=['W'], **kwargs):
        self._legal_params_ = ['W']
        
        super(Embedding, self).__init__(**kwargs)
        self._n_vocab_ = n_vocab
        self._n_out_ = n_out
        self._init_type_ = init_type
        self._W_init_ = W_init
        self._W_reg_ = W_reg
        self._trainable_params_ = trainable_params
        
    def __call__(self, in_layers):
        in_layers = to_list(in_layers)
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = in_layers[0].out_shape_ + (self._n_out_,)

        
        in_shape = in_layers[0].out_shape_
        assert len(in_shape)==2, "dim of input shape must be 2, (batch_num, n_time) Your shape is " + str(in_shape)
        input = in_layers[0].output_
        
        # init W
        if not hasattr(self, '_W_'):
            self._W_ = self._init_params(self._W_init_, self._init_type_, 
                                         shape=(self._n_vocab_, self._n_out_), 
                                         name=str(self._name_)+'_W')
        
        # set params & update reg_value
        self.set_trainable_params_and_update_reg(self._trainable_params_)
        
        # below are compulsory parts
        self._add_self_to_prevs_layer(in_layers)     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                       # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
        
    @property
    def W_(self):
        return K.get_value(self._W_)
        
    @property
    def info_(self):
        kwargs = self._base_kwargs_
        info = {'class_name': self.__class__.__name__, 
                 'n_vocab': self._n_vocab_, 
                 'n_out': self._n_out_, 
                 'init_type': self._init_type_,     # mute if W is None
                 'W': self.W_, 
                 'W_reg_info': regularizations.get_info(self._W_reg_), 
                 'trainable_params': self._trainable_params_, 
                 'kwargs': kwargs}
        return info
        
    # ---------- Public methods ----------
    
    def compile(self):
        self._inputs_ = [layer.output_ for layer in self._prevs_]
        input = K.concatenate(self._inputs_, axis=-1)
        self._output_ = self._get_output(input)
    
    @classmethod
    def load_from_info(cls, info):
        W_reg = regularizations.get_obj(info['W_reg_info'])
        
        layer = cls(n_vocab=info['n_vocab'], n_out=info['n_out'], 
                    init_type=info['init_type'], W_init=info['W'], W_reg=W_reg, 
                    trainable_params=info['trainable_params'], **info['kwargs'])
                     
        return layer
        
    # ---------- Private methods ----------
        
    # get regularization
    def _get_reg(self):
        reg_value = 0. 
        
        if self._W_reg_ is not None:
            reg_value += self._W_reg_.get_reg([self._W_])
            
        return reg_value
        
    def _get_output(self, input):
        output = self._W_[K.cast(input, 'int32')]
        return output
        
    # ------------------------------------