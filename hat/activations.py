'''
SUMMARY:  activation functions
          Pay attention if you are adding new activation, activation value 0. 
          can only occur in 0-measure set. Because Masking use 0. to judge this should be masked or not. 
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: -
--------------------------------------
'''
import numpy as np
import backend as K

def linear( x ):
    return x
    
# x can be tensor
def sigmoid( x ):
    return K.sigmoid( x )

def hard_sigmoid( x ):
    return K.hard_sigmoid( x )

# x can be tensor
def softmax( x ):
    #assert x.ndim==2    # todo
    if x.ndim==2:
        return K.softmax( x )
    if x.ndim>2:
        shape = x.shape
        x_flatten = x.reshape( ( K.prod(shape[0:-1]), shape[-1] ) )
        return K.softmax( x_flatten ).reshape( shape )
    
def tanh( x ):
    return K.tanh( x )
    
    

def relu( x, alpha=0., max_value=None ):
    return K.relu( x, alpha, max_value )
    
def zero( x ):
    return K.zeros_like(x)
    
def get( act ):
    f = globals().get( act )
    if f is None:
        raise Exception( "No this activation!" )
    else:
        return f
        
### register user defined activation
def register( act ):
    exec( act.__name__ + " = act", locals(), globals() )