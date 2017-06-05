"""
SUMMARY:  Convolutional Layers
AUTHOR:   Qiuqiang Kong
Created:  2016.05.22
Modified: 2016.05.26 Add Convolution1D (TDNN)
          2016.08.03 Add info_(), load_from_info()
--------------------------------------
"""
from core import Layer, Lambda
from ..globals import new_id
from .. import backend as K
from .. import initializations
from .. import activations
from .. import regularizations
from ..supports import to_list
import numpy as np


def dilated_filter_len(len_filter, dilation):
    return (len_filter - 1) * dilation + 1

# # Get len_out of feature maps after conv
def conv_out_len(len_in, len_filter, pad, downsample):
    len_out = (len_in - len_filter + 1 + pad * 2) // downsample
    return len_out

# Get len_out of feature maps after transpose conv
def conv_transpose_out_len(len_in, len_filter, pad, dilation):
    len_out = len_in * dilation + len_filter - 1 - pad * 2
    return len_out


class Conv1D(Layer):
    """1D Convolutional Layer. Ref: Waibel, Alexander, et al. "Phoneme recognition using time-delay neural networks." (1989)
    
    Args:
      n_outfmaps: integar. 
      len_filter: integar. 
      act: string. Nonlinearity. 
      init_type: string. Initialization type. See initializations.py for detail. 
      border_mode: integar | 'valid'. When use integar, 
          The feature map is first padded with tuple, then apply 'valid' conv2d. 
      strides: integar. 
      dilation_rate: integar. 
      W_init: ndarray. (n_outfmaps, n_in_fmaps, n_time, 1)
      b_init: ndarray. (n_outfmaps,)
      W_reg: regularization object. 
      b_reg: regularization object. 
      trainable_params: list of strings. 
      kwargs: see Layer class for details. 
    
    Input shape:
      (n_samples, n_time, n_in)
      
    Output shape:
      (n_samples, n_time, n_outfmpas)
    """
    
    def __init__(self, n_outfmaps, len_filter, act, init_type='glorot_uniform', 
                 border_mode='valid', strides=1, dilation_rate=1, W_init=None, b_init=None, 
                 W_reg=None, b_reg=None, trainable_params=['W', 'b'], **kwargs):
        self._legal_params_ = ['W', 'b']
        super(Conv1D, self).__init__(**kwargs)
        self._n_outfmaps_ = n_outfmaps
        self._len_filter_ = len_filter
        self._act_ = act
        self._init_type_ = init_type
        self._border_mode_ = border_mode
        self._strides_ = strides
        self._dilation_rate_ = dilation_rate
        self._W_init_ = W_init
        self._b_init_ = b_init
        self._W_reg_ = W_reg
        self._b_reg_ = b_reg
        self._trainable_params_ = trainable_params
        
    def __call__(self, in_layers):
        in_layers = to_list(in_layers)
        assert len(in_layers)==1, "The input of Convolution2D can only be one layer!"
        in_shape = in_layers[0].out_shape_
        assert len(in_shape)==3, "The input dimension should be 3 to Convolution2D layer!"
        self._in_shape_ = in_shape
        [batch_size, n_time, n_infmaps] = in_shape
        
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = self._get_out_shape(n_time, self._len_filter_,
                                               self._border_mode_, self._strides_, self._dilation_rate_)

        # shared params
        filter_shape = (self._n_outfmaps_, n_infmaps, self._len_filter_, 1)
        if not hasattr(self, '_W_'):
            self._W_ = self._init_params(self._W_init_, self._init_type_, 
                                        filter_shape, name=str(self._name_)+'_W')                                        
        if not hasattr(self, '_b_'):
            self._b_ = self._init_params(self._b_init_, 'zeros', 
                                        self._n_outfmaps_, name=str(self._name_)+'_b')
                                      
        # DO NOT DELETE! initialization same as keras for debug
        # tmp_shape = (self._len_filter_, 1, n_infmaps, self._n_outfmaps_)
        # self._W_ = K.shared(initializations.get(self._init_type_)(tmp_shape).transpose(3,2,0,1), name=str(self._name_)+'_W')
                                      
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
    def b_(self):
        return K.get_value(self._b_)
        
    # layer's info
    @property
    def info_(self):
        kwargs = self._base_kwargs_
        info = {'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'name': self._name_, 
                 'n_outfmaps': self._n_outfmaps_, 
                 'len_filter': self._len_filter_, 
                 'act': self._act_, 
                 'init_type': self._init_type_,
                 'border_mode': self._border_mode_, 
                 'strides': self._strides_, 
                 'dilation_rate': self._dilation_rate_, 
                 'W': self.W_, 
                 'b': self.b_, 
                 'W_reg_info': regularizations.get_info(self._W_reg_),
                 'b_reg_info': regularizations.get_info(self._b_reg_), 
                 'trainable_params': self._trainable_params_, 
                 'kwargs': kwargs}
        return info
    
    # ---------- Public methods ----------
 
    def compile(self):
        self._inputs_ =[layer.output_ for layer in self._prevs_]
        input = self._inputs_[0]
        self._output_ = self._get_output(input)
 
    @classmethod
    def load_from_info(cls, info):
        W_reg = regularizations.get_obj(info['W_reg_info'])
        b_reg = regularizations.get_obj(info['b_reg_info'])

        layer = cls(n_outfmaps=info['n_outfmaps'], len_filter=info['len_filter'], 
                     act=info['act'], init_type=info['init_type'], 
                     border_mode=info['border_mode'], strides=info['strides'], dilation_rate=info['dilation_rate'], 
                     W_init=info['W'], b_init=info['b'], W_reg=W_reg, b_reg=b_reg, trainable_params=info['trainable_params'], **info['kwargs'] )
                     
        return layer
    
    # ---------- Private methods ----------
    
    def _get_reg(self):
        reg_value = 0. 
        
        if self._W_reg_ is not None:
            if 'W' in self._trainable_params_:
                reg_value += self._W_reg_.get_reg([self._W_])
        if self._b_reg_ is not None:
            if 'b' in self._trainable_params_:
                reg_value += self._b_reg_.get_reg([self._b_])

        return reg_value
 
    def _get_pad(self, border_mode):
        if border_mode == 'valid':
            pad = 0
        # elif border_mode == 'full':
        #     pad = len_filter - 1
        elif type(border_mode) == int:
            pad = border_mode
        else:
            raise Exception("Do not support border_mode='" + border_mode + "'!")
        return pad
 
    def _get_out_shape(self, n_time, len_filter, border_mode, stride, dilation):
        len_dilated_filter = dilated_filter_len(len_filter, dilation)
        pad = self._get_pad(border_mode)
        n_time = conv_out_len(n_time, len_dilated_filter, pad, stride)
        out_shape = (None, n_time, self._n_outfmaps_)
        return out_shape
        
    def _get_output(self, input):        
        # do convolution
        input4d = input.dimshuffle(0, 2, 1, 'x')    # (batch_size, n_infmaps, n_time, 1)
        pad = self._get_pad(self._border_mode_)
        border_mode = (pad, 0)
        strides = (self._strides_, 1)
        dilation_rate = (self._dilation_rate_, 1)
        lin_out4d = K.conv2d(input4d, self._W_, border_mode=border_mode, strides=strides, dilation_rate=dilation_rate) \
                + self._b_.dimshuffle('x', 0, 'x', 'x')
        
        # output activation
        output4d = activations.get(self._act_)(lin_out4d)   # (batch_size, n_outfmaps, n_time, 1)
        output = output4d.dimshuffle(0, 2, 1, 3).flatten(3)   # (batch_size, n_time, n_outfmaps)
        return output
    
    # ------------------------------------
        

class Conv2D(Layer):
    """
    2D Convolutional Layer. 
    
    Args:
      n_outfmaps: integar. Number of feature maps. 
      n_row: integar. Height of the filter. 
      n_col: integar. Width of the filter. 
      act: string. Nonlinearity. 
      init_type: string. Initialization type. See initializations.py for detail. 
      border_mode: tuple of integars, e.g. (1,1) | 'valid'. When use tuple, 
          The feature map is first padded with tuple, then apply 'valid' conv2d. 
      strides: tuple of integar, e.g. (1,1). 
      W_init: ndarray. (n_outfmaps, n_in_fmaps, n_row, n_col)
      b_init: ndarray. (n_outfmaps,)
      W_reg: regularization object. 
      b_reg: regularization object. 
      trainable_params: list of strings. 
      kwargs: see Layer class for details. 
      
    Input shape:
      (n_samples, n_infmaps, height, width)
      
    Output shape:
      (n_samples, n_outfmaps, height, width)
    """
    def __init__(self, n_outfmaps, n_row, n_col, act, init_type='glorot_uniform', 
                  border_mode='valid', strides=(1,1), dilation_rate=(1,1), W_init=None, b_init=None, 
                  W_reg=None, b_reg=None, trainable_params=['W','b'], **kwargs):
        self._legal_params_ = ['W', 'b']
        super(Conv2D, self).__init__(**kwargs)
        self._n_outfmaps_ = n_outfmaps
        self._n_row_ = n_row
        self._n_col_ = n_col
        self._act_ = act
        self._init_type_ = init_type
        self._border_mode_ = border_mode
        self._strides_ = strides
        self._dilation_rate_ = dilation_rate
        self._W_init_ = W_init
        self._b_init_ = b_init
        self._W_reg_ = W_reg
        self._b_reg_ = b_reg
        self._trainable_params_ = trainable_params
        
    def __call__(self, in_layers):
        in_layers = to_list(in_layers)
        assert len(in_layers)==1, "The input of Convolution2D can only be one layer!"
        in_shape = in_layers[0].out_shape_
        assert len(in_shape)==4, "The input dimension should be 4 to Convolution2D layer!"
        self._in_shape_ = in_shape
        [batch_size, n_infmaps, height, width] = in_shape
        
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = self._get_out_shape(height, width, self._n_row_, self._n_col_, 
                                                self._border_mode_, self._strides_, self._dilation_rate_)
  
        # shared params
        filter_shape = (self._n_outfmaps_, n_infmaps, self._n_row_, self._n_col_)
        if not hasattr(self, '_W_'):
            self._W_ = self._init_params(self._W_init_, self._init_type_, 
                                        filter_shape, name=str(self._name_)+'_W')
        if not hasattr(self, '_b_'):                                
            self._b_ = self._init_params(self._b_init_, 'zeros', 
                                        self._n_outfmaps_, name=str(self._name_)+'_b')
                                      
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
    def b_(self):
        return K.get_value(self._b_)
        
    # layer's info
    @property
    def info_(self):
        kwargs = self._base_kwargs_
        info = {'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'name': self._name_, 
                 'n_outfmaps': self._n_outfmaps_, 
                 'n_row': self._n_row_, 
                 'n_col': self._n_col_, 
                 'act': self._act_, 
                 'init_type': self._init_type_,
                 'border_mode': self._border_mode_, 
                 'strides': self._strides_, 
                 'dilation_rate': self._dilation_rate_, 
                 'W': self.W_, 
                 'b': self.b_, 
                 'W_reg_info': regularizations.get_info(self._W_reg_),
                 'b_reg_info': regularizations.get_info(self._b_reg_), 
                 'trainable_params': self._trainable_params_, 
                 'kwargs': kwargs}
        return info
    
    # ---------- Public methods ----------
 
    def compile(self):
        self._inputs_ =[layer.output_ for layer in self._prevs_]
        input = self._inputs_[0]
        self._output_ = self._get_output(input)
 
    @classmethod
    def load_from_info(cls, info):
        W_reg = regularizations.get_obj(info['W_reg_info'])
        b_reg = regularizations.get_obj(info['b_reg_info'])

        layer = cls(n_outfmaps=info['n_outfmaps'], n_row=info['n_row'], n_col=info['n_col'], 
                     act=info['act'], init_type=info['init_type'], border_mode=info['border_mode'], 
                     strides=info['strides'], dilation_rate=info['dilation_rate'], 
                     W_init=info['W'], b_init=info['b'], W_reg=W_reg, b_reg=b_reg, trainable_params=info['trainable_params'], **info['kwargs'] )
                     
        return layer
    
    # ---------- Private methods ----------
    
    def _get_reg(self):
        reg_value = 0. 
        
        if self._W_reg_ is not None:
            if 'W' in self._trainable_params_:
                reg_value += self._W_reg_.get_reg([self._W_])
        if self._b_reg_ is not None:
            if 'b' in self._trainable_params_:
                reg_value += self._b_reg_.get_reg([self._b_])

        return reg_value
    
    def _get_pads(self, border_mode):
        if border_mode == 'valid':
            pads = (0, 0)
        elif type(border_mode) == tuple:
            pads = border_mode
        else:
            raise Exception("Do not support border_mode='" + border_mode + "'!")
        return pads
            
    def _get_out_shape(self, height, width, n_row, n_col, border_mode, strides, dilation_rate):
        dilated_row = dilated_filter_len(n_row, dilation_rate[0])
        dilated_col = dilated_filter_len(n_col, dilation_rate[1])
        pads = self._get_pads(border_mode)
        out_shape = (None, self._n_outfmaps_, 
                     conv_out_len(height, dilated_row, pads[0], strides[0]), 
                     conv_out_len(width, dilated_col, pads[1], strides[1]))
        return out_shape
        
    def _get_output(self, input):
        # do convolution
        lin_out = K.conv2d(input, self._W_, border_mode=self._border_mode_, strides=self._strides_, dilation_rate=self._dilation_rate_) \
                + self._b_.dimshuffle('x', 0, 'x', 'x')
        
        # output activation
        output = activations.get(self._act_)(lin_out)
        return output
    
    # ------------------------------------
    
    
class Conv2DTranspose(Conv2D):
    """Transpose convolution 2D. 
    
    Args:
      See Conv2D. 
    """
    def _get_out_shape(self, height, width, n_row, n_col, border_mode, strides, dilation_rate):
        pads = self._get_pads(border_mode)
        out_shape = (None, self._n_outfmaps_, 
                     conv_transpose_out_len(height, n_row, pads[0], strides[0]), 
                     conv_transpose_out_len(width, n_col, pads[1], strides[1]))
        return out_shape
        
    def _get_output(self, input):
        [batch_size, n_infmaps, height, width] = self._in_shape_
        output_shape = self._get_out_shape(height, width, self._n_row_, self._n_col_, 
                                           self._border_mode_, self._strides_, self._dilation_rate_)
                         
        lin_out = K.conv2d_transpose(input=input, 
                                     filters=self._W_.dimshuffle(1,0,2,3),    # 
                                     output_shape=output_shape, 
                                     border_mode=self._border_mode_, 
                                     strides=self._strides_) \
                + self._b_.dimshuffle('x', 0, 'x', 'x')
        
        # output activation
        output = activations.get(self._act_)(lin_out)
        return output
    
    
def _up_sampling_2d(input, **kwargs):
    assert len(kwargs['size'])==2, "Upsampling size must be 2!"
    (n_height, n_wid) = kwargs['size']
    tmp = K.repeat(input, n_height, axis=2)
    output = K.repeat(tmp, n_wid, axis=3)
    return output
   
class UpSampling2D(Lambda):
    """Up Sampling 2D. Repeat cols and rows. 
    
    Args:
      size: tuple of interger, e.g. (2,2). 
      kwargs: see Layer class for details. 
    """
    def __init__(self, size, **kwargs):
        kwargs['size'] = size
        self._size_ = size
        super(UpSampling2D, self).__init__(_up_sampling_2d, **kwargs)
    
    # ---------- Public attributes ----------
    
    @property
    def info_(self):
        kwargs = self._base_kwargs_
        info = {'class_name': self.__class__.__name__, 
                 'size': self._size_, 
                 'kwargs': kwargs}
        return info
        
    # ---------- Public methods ----------
    
    @classmethod
    def load_from_info(cls, info):
        layer = cls(info['size'], **info['kwargs'])
        return layer
    
    # ------------------------------------

    