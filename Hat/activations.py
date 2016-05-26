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
    
def sigmoid( x ):
    return K.sigmoid( x )

def softmax( x ):
    return K.softmax( x )
    
def tanh( x ):
    return K.tanh( x )
    
def relu( x, alpha=0., max_value=None ):
    return K.relu( x, alpha, max_value )
    
def get( act ):
    f = globals().get( act )
    if f is None:
        raise Exception( "No this activation!" )
    else:
        return f