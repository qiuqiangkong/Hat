"""
SUMMARY:  activation functions
AUTHOR:   Qiuqiang Kong
Created:  2016.05.11
Modified: -
--------------------------------------
"""
import numpy as np
import backend as K

def linear( x ):
    return x
    
def sigmoid( x ):
    """x can be tensor
    """
    return K.sigmoid( x )

def hard_sigmoid( x ):
    return K.hard_sigmoid( x )

def softmax( x ):
    """x can be tensor
    """
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
    
def leaky_relu( x, alpha=0.2, max_value=None ):
    return K.relu( x, alpha, max_value )
    
def zero( x ):
    return K.zeros_like(x)
  
def get( act ):
    f = globals().get( act )
    if f is None:
        raise Exception( "No this activation!" )
    else:
        return f
    
        
def register( act ):
    """Register user defined activation. 
    """
    exec( act.__name__ + " = act", locals(), globals() )