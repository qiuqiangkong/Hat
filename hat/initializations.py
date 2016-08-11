'''
SUMMARY:  initializations of weights
AUTHOR:   Qiuqiang Kong
Created:  2016.05.20
Modified: 2016.08.02 modify all return to non shared
--------------------------------------
'''
import numpy as np
import backend as K

'''
zeros tensor
'''
def zeros( shape ):
    return np.zeros( shape )
    
'''
ones tensor
'''
def ones( shape ):
    return np.ones( shape )

'''
identity matrix
'''
def eye( len ):
    return np.eye( len )

'''
uniform tensor
'''
def uniform( shape, scale=0.05 ):
    return np.random.uniform( -scale, scale, shape )
    
'''
glorot uniform
[1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." 2010.
'''
def glorot_uniform( shape )   :
    scale = np.sqrt( 6. / sum( shape ) )
    return uniform( shape, scale )


### return function from name
def get( init ):
    f = globals().get( init )
    if f is None:
        raise Exception( "No this init_value method!" )
    else:
        return f