import backend as K
import numpy as np

_EPSILON = 1e-6     # when set to 1e-8, binary_crossentropy underflow

# categorical crossentropy
def categorical_crossentropy( p_y_pred, y_gt ):
    p_y_pred = K.clip( p_y_pred, _EPSILON, 1. - _EPSILON )
    return K.categorical_crossentropy( p_y_pred, y_gt )
    
# binary crossentropy
def binary_crossentropy( p_y_pred, y_gt ):
    p_y_pred = K.clip( p_y_pred, _EPSILON, 1. - _EPSILON )    
    return K.binary_crossentropy( p_y_pred, y_gt )
    
# mean square error
def mse( p_y_pred, y_gt ):
    return K.mean( K.sum( K.sqr( p_y_pred - y_gt ), axis=-1 ) )
    
#todo not well tested
def kl_divergence( y_pred, y_gt ):
    y_pred = K.clip( y_pred, _EPSILON, 1. - _EPSILON )
    y_gt = K.clip( y_gt, _EPSILON, 1. - _EPSILON )
    return K.mean( K.sum( y_gt * K.log( y_gt / y_pred ) - y_gt + y_pred, axis=-1 ) )
    
    
def get( loss ):
    f = globals().get( loss )
    if f is None:
        
        raise Exception( 'No ' + loss + ' loss!' )
    else:
        return f