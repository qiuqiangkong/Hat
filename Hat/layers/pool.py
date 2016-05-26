from core import Layer, Lambda
from ..globals import new_id
from .. import backend as K
from .. import initializations
from .. import activations
from ..supports import to_list, get_mask
import numpy as np
from theano.tensor.signal import downsample

# for cnn
def _max_pool_2d( input, in_shape, **kwargs ):
    assert len(in_shape)==4     # (batch_size, n_infmaps, height, width)
    
    # init kwargs
    [batch_size, n_infmaps, height, width] = in_shape
    pool_size = kwargs['pool_size']
    
    # downsample
    output = downsample.max_pool_2d( input, pool_size, ignore_border=True )
    out_shape = ( None, n_infmaps, int(height/pool_size[0]), int(width/pool_size[1]) )
    return output, out_shape
    
class MaxPool2D( Lambda ):
    def __init__( self, name=None, **kwargs ):
        assert 'pool_size' in kwargs, "You must specifiy pool_size kwarg in MaxPool2D!"
        super( MaxPool2D, self ).__init__( _max_pool_2d, name, **kwargs )

'''
Mean along time axis in RNN. 
'''
def _global_mean_time_pool( input, in_shape, **kwargs ):
    assert len(in_shape)==3, "Input dimension must be 3, (batch_size, n_time, n_freq)"
    masking = kwargs['masking']
    
    if masking is True:
        output = K.sum( input, axis=1 )
        mask = get_mask( input )
        batch_nums = K.sum( mask, axis=1 )
        output /= batch_nums[:, None]
    else:
        output = K.mean( input, axis=1 )
        
    out_shape = ( in_shape[0], in_shape[2] )
    return output, out_shape
    
class GlobalMeanTimePool( Lambda ):
    def __init__( self, name=None, **kwargs ):
        assert 'masking' in kwargs, "You must specifiy masking kwarg in GlobalMeanTimePool!"
        super( GlobalMeanTimePool, self ).__init__( _global_mean_time_pool, name, **kwargs )
        

'''
max pool along axis
'''
def _global_max_pool( input, in_shape, **kwargs ):
    axis = kwargs['axis']
    output = K.max( input, axis )
    out_shape = in_shape[0:axis] + in_shape[axis+1:]
    return output, out_shape
    
class GlobalMaxPool( Lambda ):
    def __init__( self, name=None, **kwargs ):
        assert 'axis' in kwargs, "You must specifiy axis kwarg in GlobalMaxPool!"
        super( GlobalMaxPool, self ).__init__( _global_max_pool, name, **kwargs )