'''
SUMMARY:  Convolutional Layers
AUTHOR:   Qiuqiang Kong
Created:  2016.05.22
Modified: 2016.05.26 Add Convolution1D (TDNN)
--------------------------------------
'''
from core import Layer
from ..globals import new_id
from .. import backend as K
from .. import initializations
from .. import activations
from ..supports import to_list
import numpy as np


# Get len_out of feature maps after conv
def conv_out_len( len_in, len_filter, border_mode ):
    if border_mode=='same':
        len_out = len_in
    elif border_mode=='valid':
        len_out = len_in - len_filter + 1
    elif border_mode=='full':
        len_out = len_in + len_filter - 1
    else:
        raise Exception("Do not support border_mode='" + border_mode + "'!")
    return len_out

'''
Ref: Waibel, Alexander, et al. "Phoneme recognition using time-delay neural networks." 
Acoustics, Speech and Signal Processing, IEEE Transactions on 37.3 (1989): 328-339.
'''

class Convolution1D( Layer ):
    def __init__( self, n_outfmaps, len_filter, act, init_type='uniform', reg=None, name=None ):
        super( Convolution1D, self ).__init__( name )
        self._n_outfmaps = n_outfmaps
        self._len_filter = len_filter
        self._act = act
        self._init_type = init_type
        self._reg = reg
        
    def __call__( self, in_layers ):
        # only one input layer is allowed
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Dense can only be one layer!"
        in_layer = in_layers[0]
        
        # input dim must be 3
        in_shape = in_layer.out_shape
        input = in_layer.output
        assert len(in_shape)==3, "Input shape should be (batch_size, n_time, n_in), yours is " + str(in_shape)
        [batch_size, n_time, n_in] = in_shape
        
        # do convolution
        filter_shape = ( self._n_outfmaps, 1, self._len_filter, n_in )
        self._W = initializations.get( self._init_type )( filter_shape, name=str(self._name)+'_W' )
        self._b = initializations.zeros( self._n_outfmaps, name=str(self._name)+'_b' )
        
        # shape(lin_out): (batch_size, n_outfmaps, n_time, 1)
        lin_out = K.conv2d( input.dimshuffle(0,'x',1,2), self._W, border_mode='valid' ) + self._b.dimshuffle('x', 0, 'x', 'x')
        
        # shape(lin_out): (batch_size, n_time, n_outfmaps)
        lin_out = lin_out.dimshuffle(0,2,1,3).flatten(3)
        
        # activation function
        output = activations.get( self._act )( lin_out )
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = ( None, conv_out_len(n_time, self._len_filter, border_mode='valid'), self._n_outfmaps )
        self._output = output
        self._params = [ self._W, self._b ]
        self._reg_value = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self

    # get regularization
    def _get_reg( self ):
        if self._reg is None:
            _reg_value = 0.
        else:
            _reg_value = self._reg.reg_value( [ self._W ] )
        return _reg_value
        
    def W():
        return self._W
        
    def b():
        return self._b

class Convolution2D( Layer ):
    def __init__( self, n_outfmaps, n_row, n_col, act, init_type='uniform', border_mode='full', reg=None, name=None ):
        super( Convolution2D, self ).__init__( name )
        self._n_outfmaps = n_outfmaps
        self._n_row = n_row
        self._n_col = n_col
        self._act = act
        self._init_type = init_type
        self._border_mode = border_mode
        self._reg = reg
        
    def __call__( self, in_layers ):
        # only one input layer is allowed
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Dense can only be one layer!"
        in_layer = in_layers[0]
        
        # input dim must be 4
        in_shape = in_layer.out_shape
        input = in_layer.output
        assert len(in_shape)==4     # (batch_size, n_infmaps, height, width)
        [batch_size, n_infmaps, height, width] = in_shape
        
        # do convolution
        filter_shape = ( self._n_outfmaps, n_infmaps, self._n_row, self._n_col )
        self._W = initializations.get( self._init_type )( filter_shape, name=str(self._name)+'_W' )
        self._b = initializations.zeros( self._n_outfmaps, name=str(self._name)+'_b' )
        lin_out = K.conv2d( input, self._W, border_mode=self._border_mode ) + self._b.dimshuffle('x', 0, 'x', 'x')
        output = activations.get( self._act )( lin_out )
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = ( None, self._n_outfmaps, 
                            conv_out_len(height, self._n_row, self._border_mode), 
                            conv_out_len(width, self._n_col, self._border_mode) )
        self._output = output
        self._params = [ self._W, self._b ]
        self._reg_value = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    # get regularization
    def _get_reg( self ):
        if self._reg is None:
            _reg_value = 0.
        else:
            _reg_value = self._reg.reg_value( [ self._W ] )
        return _reg_value
        
    def W():
        return self._W
        
    def b():
        return self._b