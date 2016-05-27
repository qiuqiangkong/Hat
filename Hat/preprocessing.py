'''
SUMMARY:  preprocessing data
          move these files from supports to this file
AUTHOR:   Qiuqiang Kong
Created:  2016.05.25
Modified: 2016.05.27 Modify pad_trunc_seqs
--------------------------------------
'''
import numpy as np
from supports import to_list

# truncate seq or pad with 0, input can be list or np.ndarray
# the element in x can be list or ndarray, then pad or trunc all elements in x to max_len
def pad_trunc_seqs( x, max_len ):
    type_x = type( x )
    N = len( x )
    list_new = []
    for e in x:
        L = len( e )
        if type(e)==np.ndarray:
            shape = e.shape
            if L < max_len:
                pad_shape = (max_len-L,) + shape[1:]
                pad = np.zeros( pad_shape )
                list_new.append( np.vstack( (e, pad) ) )
            else:
                list_new.append( e[0:max_len] )
        if type(e)==list:
            if L < max_len:
                pad = [0] * ( max_len - L )
                list_new.append( e + pad )
            else:
                list_new.append( e[0:max_len] )
    
    if type_x==list:
        return list_new
    if type_x==np.ndarray:
        return np.array( list_new )
    

# convert from 3d to 4d, for input of cnn        
def reshape_3d_to_4d( X ):
    [ N, n_row, n_col ] = X.shape
    return X.reshape( (N, 1, n_row, n_col) )
    
# sparse label to categorical label
def sparse_to_categorical( x, n_out ):
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros( (N,n_out) )
    x_categ[ np.arange(N), x ] = 1
    return x_categ.reshape( (shape)+(n_out,) )