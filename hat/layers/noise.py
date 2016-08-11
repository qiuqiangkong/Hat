'''
SUMMARY:  Add noise. Currently supporting normal distribution
AUTHOR:   Qiuqiang Kong
Created:  2016.06.02
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

class AddNoise( Layer ):
    def __init__( self, random_type, name=None, **kwargs ):
        super( AddNoise, self ).__init__( name )
        self._random_type = random_type
        self._kwargs = kwargs
        self._tr_phase_node = K.common_tr_phase_node
        
    def __call__( self, in_layers ):
        # only one input layer is allowed
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Dense can only be one layer!"
        in_layer = in_layers[0]
        
        # add random noise
        input = in_layer.output
        output = K.ifelse( K.eq( self._tr_phase_node, 1. ), self._tr_phase(input), input )
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = in_layer.out_shape
        self._output = output
        self._params = []
        self._reg_value = 0.
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    def _tr_phase( self, input ):
        if self._random_type=='normal': 
            assert 'avg' in self._kwargs, "You must specifiy avg using normal in AddNoise!"
            assert 'std' in self._kwargs, "You must specifiy std using normal in AddNoise!"
            output = input + K.rng_normal( input.shape, self._kwargs['avg'], self._kwargs['std'] )
        return output