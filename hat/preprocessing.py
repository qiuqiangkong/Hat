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


def pad_trunc_seqs(x, max_len, pad_type='post'):
    """Pad or truncate sequences. 
    
    Args:
      x: ndarray | list of ndarray. Each element in x should be ndarray. Each
          element in x is padded with 0 or truncated to max_len. 
      max_len: int, length to be padded with 0 or truncated. 
      pad_type, string, 'pre' | 'post'. 
      
    Returns: 
      x_new: ndarray, (N, ndarray), padded or truncated sequences. 
      mask: ndarray, (N, max_len), mask of padding. 
    """
    list_x_new, list_mask = [], []
    for e in x:
        L = len(e)
        e_new, mask = pad_trunc_seq(e, max_len, pad_type)
        list_x_new.append(e_new)
        list_mask.append(mask)
    
    type_x = type(x)
    if type_x==list:
        return list_x_new, list_mask
    elif type_x==np.ndarray:
        return np.array(list_x_new), np.array(list_mask)
    else:
        raise Exception("Input should be list or ndarray!")
    

def pad_trunc_seq(x, max_len, pad_type='post'):
    """Pad or truncate ndarray. 
    
    Args:
      x: ndarray. 
      max_len: int, length to be padded with 0 or truncated. 
      pad_type, string, 'pre' | 'post'. 
      
    Returns:
      x_new: ndarray, padded or truncated ndarray. 
      mask: 1d-array, mask of padding. 
    """
    L = len(x)
    shape = x.shape
    data_type = x.dtype
    if L < max_len:
        pad_shape = (max_len-L,) + shape[1:]
        pad = np.zeros(pad_shape)
        if pad_type=='pre': 
            x_new = np.concatenate((pad, x), axis=0)
            mask = np.concatenate([np.zeros(max_len-L), np.ones(L)])
        elif pad_type=='post': 
            x_new = np.concatenate((x, pad), axis=0)
            mask = np.concatenate([np.ones(L), np.zeros(max_len-L)])
        else:
            raise Exception("pad_type should be 'post' | 'pre'!")
    else:
        if pad_type=='pre':
            x_new = x[L-max_len:]
            mask = np.ones(max_len)
        elif pad_type=='post':
            x_new = x[0:max_len]
            mask = np.ones(max_len)
        else:
            raise Exception("pad_type should be 'post' | 'pre'!")
    x_new = x_new.astype(data_type)
    return x_new, mask

# enframe sequence to matrix
def enframe(x, win, inc):
    Xlist = []
    p = 0
    while (p+win <= len(x)):
        Xlist.append(x[p:p+win])
        p += inc
    
    X = np.array(Xlist)
    return X

def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments. 
    
    Args:
      x: 2darray, (n_time, n_in)
      agg_num: int, number of frames to concatenate. 
      hop: int, number of hop frames. 
      
    Returns:
      3darray, (n_blocks, agg_num, n_in)
    """
    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)

# convert from 3d to 4d, for input of cnn        
def reshape_3d_to_4d(X):
    [N, n_row, n_col] = X.shape
    return X.reshape((N, 1, n_row, n_col))
    
    
# sparse label to categorical label
# x: ndarray
def sparse_to_categorical(x, n_out):
    x = x.astype(int)   # force type to int
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros((N,n_out))
    x_categ[np.arange(N), x] = 1
    return x_categ.reshape((shape)+(n_out,))
