'''
SUMMARY:  preprocessing data
          move from supports to this file
AUTHOR:   Qiuqiang Kong
Created:  2016.05.25
Modified: -
--------------------------------------
'''
import numpy as np

# truncate seq or pad with 0, input can be list or np.ndarray
# if x is list, pad or trunc all elements in x to max_len
# if x is ndarray, pad or trunc it to max_len
def pad_trunc_seqs( x, max_len ):
    type_x = type( x )
    list_x = to_list( x )
    list_new = []
    for e in list_x:
        shape = e.shape
        N = len( e )
        if N < max_len:
            pad_shape = (max_len-N,) + shape[1:]
            pad = np.zeros( pad_shape )
            list_new.append( np.vstack( (e, pad) ) )
        else:
            list_new.append( e[0:max_len] )
    if type_x==list:
        return list_new
    if type_x==np.ndarray:
        return list_new[0]

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