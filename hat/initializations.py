"""
SUMMARY:  initializations of weights
AUTHOR:   Qiuqiang Kong
Created:  2016.05.20
Modified: 2017.02.19
--------------------------------------
"""
import numpy as np
import backend as K


def zeros( shape ):
    return np.zeros( shape )
    
def ones( shape ):
    return np.ones( shape )

def eye( len ):
    """Identity matrix
    """
    return np.eye( len )

def uniform( shape, scale=0.01 ):
    return np.random.uniform( -scale, scale, shape )
    

def glorot_uniform( shape ):   
    """
    glorot uniform
    [1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." 2010.
    """ 
    if len(shape)==2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape)==4:
        receptive_field_size = shape[2]*shape[3]
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    scale = np.sqrt( 6. / ( fan_in+fan_out ) )
    return uniform( shape, scale )


### return function from name
def get( init ):
    f = globals().get( init )
    if f is None:
        raise Exception( "No this init_value method!" )
    else:
        return f