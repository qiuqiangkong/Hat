'''
SUMMARY:  Convolutional Layers
AUTHOR:   Qiuqiang Kong
Created:  2016.05.22
Modified: 2016.05.26 Add Convolution1D (TDNN)
          2016.08.03 Add info_(), load_from_info()
--------------------------------------
'''
from core import Layer, Lambda
from ..globals import new_id
from .. import backend as K
from .. import initializations
from .. import activations
from .. import regularizations
from ..supports import to_list
import numpy as np


# Get len_out of feature maps after conv
def conv_out_len( len_in, len_filter, border_mode, downsample ):
    if border_mode=='valid':
        len_out = (len_in - len_filter + 1) // downsample
    elif border_mode=='full':
        len_out = (len_in + len_filter - 1) // downsample
    else:
        raise Exception("Do not support border_mode='" + border_mode + "'!")
    return len_out

'''
[1] Waibel, Alexander, et al. "Phoneme recognition using time-delay neural networks." (1989)
input shape should be (n_samples, n_fmaps, n_time)
border_mode: 'valid' | 'full' | #a1 (pad with this #a1 then do valid conv)
'''
class Convolution1D( Layer ):
    def __init__( self, n_outfmaps, len_filter, act, init_type='glorot_uniform', border_mode='valid', stride=1,
                  W_init=None, b_init=None, W_reg=None, b_reg=None, name=None ):
        super( Convolution1D, self ).__init__( name )
        self._n_outfmaps_ = n_outfmaps
        self._len_filter_ = len_filter
        self._act_ = act
        self._init_type_ = init_type
        self._border_mode_ = border_mode    # should be fixed to 'valid'
        self._stride_ = stride
        self._W_init_ = W_init
        self._b_init_ = b_init
        self._W_reg_ = W_reg
        self._b_reg_ = b_reg
        
    def __call__( self, in_layers ):
        # only one input layer is allowed
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Dense can only be one layer!"
        in_layer = in_layers[0]
        in_shape = in_layers[0].out_shape_
        
        # input dim must be 3
        input = in_layer.output_
        assert len(in_shape)==3, "Input shape should be (batch_size, n_time, n_in), yours is " + str(in_shape)
        [batch_size, n_infmaps, n_time] = in_shape
        
        # do convolution
        filter_shape = ( self._n_outfmaps_, n_infmaps, self._len_filter_, 1 )
        self._W_ = self._init_params( self._W_init_, self._init_type_, filter_shape, name=str(self._name_)+'_W' )
        self._b_ = self._init_params( self._b_init_, 'zeros', self._n_outfmaps_, name=str(self._name_)+'_b' )
        
        # shape(lin_out): (batch_size, n_outfmaps, n_time, 1)
        border_mode = self._get_border_mode( self._border_mode_ )
        lin_out = K.conv2d( input.dimshuffle(0,1,2,'x'), self._W_, border_mode, strides=(self._stride_,1) ) + self._b_.dimshuffle('x', 0, 'x', 'x')
        
        # shape(lin_out): (batch_size, n_outfmaps, n_time)
        lin_out = lin_out.dimshuffle(0,1,2,3).flatten(3)
        
        # activation function
        output = activations.get( self._act_ )( lin_out )
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = self._get_out_shape( n_time, self._len_filter_, border_mode, self._stride_ )
        self._output_ = output
        self._params_ = [ self._W_, self._b_ ]
        self._reg_value_ = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                             # check if all attributes are implemented
        return self

    # ---------- Public attributes ----------
    
    @property    
    def W_( self ):
        return K.get_value( self._W_ )
        
    @property
    def b_( self ):
        return K.get_value( self._b_ )
        
    # layer's info
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'name': self._name_, 
                 'n_outfmaps': self._n_outfmaps_, 
                 'len_filter': self._len_filter_, 
                 'act': self._act_, 
                 'init_type': self._init_type_,
                 'border_mode': self._border_mode_, 
                 'stride': self._stride_, 
                 'W': self.W_, 
                 'b': self.b_, 
                 'W_reg_info': regularizations.get_info( self._W_reg_ ),
                 'b_reg_info': regularizations.get_info( self._b_reg_ ), }
        return dict
    
    # ---------- Public methods ----------
    
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        W_reg = regularizations.get_obj( info['W_reg_info'] )
        b_reg = regularizations.get_obj( info['b_reg_info'] )

        layer = cls( n_outfmaps=info['n_outfmaps'], len_filter=info['len_filter'], 
                     act=info['act'], init_type=info['init_type'], 
                     W_init=info['W'], b_init=info['b'], W_reg=W_reg, b_reg=b_reg, name=info['name'] )
                     
        return layer
    
    # ---------- Private methods ----------
    
    # get regularization
    def _get_reg( self ):
        reg_value = 0. 
        
        if self._W_reg_ is not None:
            reg_value += self._W_reg_.get_reg( [self._W_] )
            
        if self._b_reg_ is not None:
            reg_value += self._b_reg_.get_reg( [self._b_] )
            
        return reg_value
    
    # get border_mode
    def _get_border_mode( self, border_mode ):
        if type(self._border_mode_) is int: 
            return (self._border_mode_,1)
        else:
            return self._border_mode_
    
    # get out shape
    def _get_out_shape( self, n_time, len_filter, border_mode, stride ):
        if border_mode=='valid' or border_mode=='full':
            out_shape = ( None, self._n_outfmaps_, 
                                conv_out_len(n_time, len_filter, border_mode, stride) )
        elif type(border_mode) is tuple:
            pad_n_time = height + border_mode[0] * 2
            out_shape = ( None, self._n_outfmaps_, 
                                conv_out_len(pad_n_time, len_filter, 'valid', stride) )
        return out_shape
    
    # ------------------------------------



'''
2D Convolutional Layer
input shape should be (n_samples, n_fmaps, height, width)
border_mode: 'valid' | 'full' | (a1, a2) (pad with this tuple then do valid conv)
'''
class Convolution2D( Layer ):
    def __init__( self, n_outfmaps, n_row, n_col, act, init_type='glorot_uniform', border_mode='valid', strides=(1,1),
                  W_init=None, b_init=None, W_reg=None, b_reg=None, name=None ):
        super( Convolution2D, self ).__init__( name )
        self._n_outfmaps_ = n_outfmaps
        self._n_row_ = n_row
        self._n_col_ = n_col
        self._act_ = act
        self._init_type_ = init_type
        self._border_mode_ = border_mode
        self._strides_ = strides
        self._W_init_ = W_init
        self._b_init_ = b_init
        self._W_reg_ = W_reg
        self._b_reg_ = b_reg
        
    def __call__( self, in_layers ):
        # only one input layer is allowed
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Dense can only be one layer!"
        in_layer = in_layers[0]
        
        # input dim must be 4
        in_shape = in_layer.out_shape_
        input = in_layer.output_
        assert len(in_shape)==4     # (batch_size, n_infmaps, height, width)
        [batch_size, n_infmaps, height, width] = in_shape
        
        # init params
        filter_shape = ( self._n_outfmaps_, n_infmaps, self._n_row_, self._n_col_ )
        self._W_ = self._init_params( self._W_init_, self._init_type_, filter_shape, name=str(self._name_)+'_W' )
        self._b_ = self._init_params( self._b_init_, 'zeros', self._n_outfmaps_, name=str(self._name_)+'_b' )

        # do convolution
        lin_out = K.conv2d( input, self._W_, border_mode=self._border_mode_, strides=self._strides_ ) + self._b_.dimshuffle('x', 0, 'x', 'x')
        
        # output activation
        output = activations.get( self._act_ )( lin_out )
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = self._get_out_shape( height, width, self._n_row_, self._n_col_, self._border_mode_, self._strides_ )
        self._output_ = output
        self._params_ = [ self._W_, self._b_ ]
        self._reg_value_ = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                             # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
    
    @property    
    def W_( self ):
        return K.get_value( self._W_ )
        
    @property
    def b_( self ):
        return K.get_value( self._b_ )
        
    # layer's info
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'name': self._name_, 
                 'n_outfmaps': self._n_outfmaps_, 
                 'n_row': self._n_row_, 
                 'n_col': self._n_col_, 
                 'act': self._act_, 
                 'init_type': self._init_type_,
                 'border_mode': self._border_mode_, 
                 'W': self.W_, 
                 'b': self.b_, 
                 'W_reg_info': regularizations.get_info( self._W_reg_ ),
                 'b_reg_info': regularizations.get_info( self._b_reg_ ), }
        return dict
    
    # ---------- Public methods ----------
    
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        W_reg = regularizations.get_obj( info['W_reg_info'] )
        b_reg = regularizations.get_obj( info['b_reg_info'] )

        layer = cls( n_outfmaps=info['n_outfmaps'], n_row=info['n_row'], n_col=info['n_col'], 
                     act=info['act'], init_type=info['init_type'], border_mode=info['border_mode'], 
                     W_init=info['W'], b_init=info['b'], W_reg=W_reg, b_reg=b_reg, name=info['name'] )
                     
        return layer
    
    # ---------- Private methods ----------
    
    # get regularization
    def _get_reg( self ):
        reg_value = 0. 
        
        if self._W_reg_ is not None:
            reg_value += self._W_reg_.get_reg( [self._W_] )
            
        if self._b_reg_ is not None:
            reg_value += self._b_reg_.get_reg( [self._b_] )
            
        return reg_value
    
    # get out shape
    def _get_out_shape( self, height, width, n_row, n_col, border_mode, strides ):
        if border_mode=='valid' or border_mode=='full':
            out_shape = ( None, self._n_outfmaps_, 
                                conv_out_len(height, n_row, border_mode, strides[0]), 
                                conv_out_len(width, n_col, border_mode, strides[1]) )
        elif type(border_mode) is tuple:
            pad_height = height + border_mode[0] * 2
            pad_width = width + border_mode[1] * 2
            out_shape = ( None, self._n_outfmaps_, 
                                conv_out_len(pad_height, n_row, 'valid', strides[0]), 
                                conv_out_len(pad_width, n_col, 'valid', strides[1]) )
        return out_shape
    
    # ------------------------------------
    
   

def _zero_padding_2d( input, **kwargs ):
    padding = kwargs['padding']
    in_shape = input.shape
    
    # out shape after padding
    out_shape = list(in_shape)
    out_shape[-2] = in_shape[-2] + 2*padding[0]
    out_shape[-1] = in_shape[-1] + 2*padding[1]
    
    # assign value
    output = K.zeros(out_shape)
    slices = [slice(None)] * (input.ndim-2) + [ slice( padding[0], out_shape[-2]-padding[0] ) ] \
           + [ slice( padding[1], out_shape[-1]-padding[1] ) ]
    output = K.set_subtensor( output[slices],input )
    return output
   
class ZeroPadding2D( Lambda ):
    def __init__( self, name=None, **kwargs ):
        assert 'padding' in kwargs, "You must specifiy 'padding' kwarg in ZeroPadding2D!"
        super( ZeroPadding2D, self ).__init__( _zero_padding_2d, name, **kwargs )

           
    