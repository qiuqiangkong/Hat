from core import Layer
from ..globals import new_id
from .. import backend as K
from .. import initializations
from .. import activations
from ..supports import to_list
import numpy as np
from theano.tensor.nnet import conv2d

class Convolution2D( Layer ):
    def __init__( self, n_outfmaps, n_row, n_col, act, init_type='uniform', reg=None, name=None ):
        super( Convolution2D, self ).__init__( name )
        self._n_outfmaps = n_outfmaps
        self._n_row = n_row
        self._n_col = n_col
        self._act = act
        self._init_type = init_type
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
        ''' border_mode: 'valid', len_data - len_filter + 1
                         'full', len_data + len_filter - 1
        '''
        filter_shape = ( self._n_outfmaps, n_infmaps, self._n_row, self._n_col )
        self._W = initializations.get( self._init_type )( filter_shape, name=str(self._name)+'_W' )
        self._b = initializations.zeros( self._n_outfmaps, name=str(self._name)+'_b' )
        lin_out = conv2d( input, self._W, border_mode='valid' ) + self._b.dimshuffle('x', 0, 'x', 'x')
        output = activations.get( self._act )( lin_out )
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = ( None, self._n_outfmaps, height-self._n_row+1, width-self._n_col+1 )
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