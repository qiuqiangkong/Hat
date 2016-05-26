import numpy as np
import backend as K

# All initialization are used for shared variable

def zeros( shape, name=None ):
    return K.sh_variable( value=np.zeros(shape), name=name )
    
def ones( shape, name=None ):
    return K.sh_variable( value=np.ones(shape), name=name )

def eye( len, name=None ):
    return K.sh_variable( value=np.eye(len), name=name )

def uniform( shape, scale=0.05, name=None ):
    
    return K.sh_variable( value=np.random.uniform( -scale, scale, shape ), name=name )
    
def glorot_uniform( shape, name=None ):
    scale = np.sqrt( 6. / sum( shape ) )
    return uniform( shape, scale, name )
    
def get( init ):
    f = globals().get( init )
    if f is None:
        raise Exception( "No this init_value method!" )
    else:
        return f