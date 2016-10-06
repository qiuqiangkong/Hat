'''
SUMMARY:  Example for imdb classification, using Lenet-CNN
          Training time: 25 s/epoch. (Tesla M2090)
          Test error: 0.74% after 30 epoches. (Better results can be got by tuning hyper-params)
AUTHOR:   Qiuqiang Kong
Created:  2016.05.25
Modified: 2016.05.27 Modify pad_trunc_seqs
          2016.05.28 Add force int(x) in sparse_to_categorical
          2016.05.30 Fix bug in pad_trunc_seqs
          2016.06.25 Add pad_trunc_seq
--------------------------------------
'''
import numpy as np
from supports import to_list

# truncate seq or pad with 0, input can be list or np.ndarray
# the element in x can be list or ndarray, then pad or trunc all elements in x to max_len
# type: 'post' | 'pre'
def pad_trunc_seqs( x, max_len, pad_type='post' ):
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
                if pad_type=='pre': list_new.append( np.concatenate( (pad, e), axis=0 ) )
                if pad_type=='post': list_new.append( np.concatenate( (e, pad), axis=0 ) )
            else:
                if pad_type=='pre': list_new.append( e[L-max_len:] )
                if pad_type=='post': list_new.append( e[0:max_len] )
        if type(e)==list:
            if L < max_len:
                pad = [0] * ( max_len - L )
                if pad_type=='pre': list_new.append( pad + e )
                if pad_type=='post': list_new.append( e + pad )
            else:
                if pad_type=='pre': list_new.append( e[L-max_len:] )
                if pad_type=='post': list_new.append( e[0:max_len] )
    
    if type_x==list:
        return list_new
    if type_x==np.ndarray:
        return np.array( list_new )
        
# pad or trunc seq, x should be ndarray
def pad_trunc_seq( x, max_len, pad_type='post' ):
    L = len(x)
    shape = x.shape
    if L < max_len:
        pad_shape = (max_len-L,) + shape[1:]
        pad = np.zeros( pad_shape )
        if pad_type=='pre': return np.concatenate( (pad, x), axis=0 )
        if pad_type=='post': return np.concatenate( (x, pad), axis=0 )
    else:
        if pad_type=='pre': return x[L-max_len:]
        if pad_type=='post': return x[0:max_len]

# enframe sequence to matrix
def enframe( x, win, inc ):
    Xlist = []
    p = 0
    while ( p+win <= len(x) ):
        Xlist.append( x[p:p+win] )
        p += inc
    
    X = np.array( Xlist )
    return X

# concatenate feautres     
def mat_2d_to_3d( X, agg_num, hop ):
    # pad to at least one block
    len_X, n_in = X.shape
    if ( len_X < agg_num ):
        X = np.concatenate( ( X, np.zeros((agg_num-len_X, n_in)) ) )
        
    # agg 2d to 3d
    len_X = len( X )
    i1 = 0
    X3d = []
    while ( i1+agg_num <= len_X ):
        X3d.append( X[i1:i1+agg_num] )
        i1 += hop
    return np.array( X3d )

# convert from 3d to 4d, for input of cnn        
def reshape_3d_to_4d( X ):
    [ N, n_row, n_col ] = X.shape
    return X.reshape( (N, 1, n_row, n_col) )
    
    
# sparse label to categorical label
# x is 1-dim ndarray
def sparse_to_categorical( x, n_out ):
    x = x.astype(int)   # force type to int
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros( (N,n_out) )
    x_categ[ np.arange(N), x ] = 1
    return x_categ.reshape( (shape)+(n_out,) )