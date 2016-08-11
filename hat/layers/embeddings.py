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
from .. import activations
from ..supports import to_list
import numpy as np

class Embedding( Layer ):
    def __init__( self, n_vocab, n_out, init_type='uniform', reg=None, name=None ):
        super( Embedding, self ).__init__( name )
        self._n_vocab = n_vocab
        self._n_out = n_out
        self._init_type = init_type
        self._reg = reg
        
    def __call__( self, in_layers ):
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Embedding can only be one layer!"
        in_layer = in_layers[0]
        
        in_shape = in_layer.out_shape
        assert len(in_shape)==2, "dim of input shape must be 2, (batch_num, n_time) Your shape is " + str(in_shape)
        input = in_layer.output
        
        # init W
        self._W = initializations.get( self._init_type )( (self._n_vocab, self._n_out), name=str(self._name)+'_W' )
        
        # output 
        output = self._W[ K.cast( input, 'int32' ) ]
        self.tmp = K.cast( input, 'int32' )
        
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = in_shape + (self._n_out,)
        print self._out_shape
        self._output = output
        self._params = [ self._W ]
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
            _reg_value = self._reg.reg_value( [ self.W ] )
        return _reg_value
        
    @property
    def W( self ):
        return self._W