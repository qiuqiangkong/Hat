"""
SUMMARY:  support functions
AUTHOR:   Qiuqiang Kong
Created:  2016.05.17
Modified: -
--------------------------------------
"""
import numpy as np
import sys
import time
import backend as K

### basic
# data to tuple
def to_tuple(x):
    if type(x) is not tuple:
        x = (x,)
    return x
    
# data to list
def to_list(x):
    if type(x) is not list:
        x = [x]
    return x
    
# format data list
def format_data_list(x):
    return [K.format_data(e) for e in x]
    
def is_one_element(x):
    if len(x)==1: return True
    else: return False
    
def is_elem_equal(x):
    return len(set(x)) <= 1
    
# Get mask from data (for RNN). size(mask): batch_num*n_time
def get_mask(input):
    return K.neq(K.sum(K.abs(input), axis=-1), 0.)
    
### data related
# shuffle data
def shuffle(xs, ys):
    N = len(xs[0])
    assert all(len(e)==N for e in xs) and all(len(e)==N for e in ys), "all the length of elements in xs and ys should be equal!"
    idx = np.arange(N)
    np.random.shuffle(idx)
    
    xs = [x[idx] for x in xs]
    ys = [y[idx] for y in ys]
    
    return xs, ys
    
# x, y, mask should be list of ndarray of same batch_num
def shuffle_xymask(x, y, mask):
    N = len(x[0])
    idx = np.arange(N)
    np.random.shuffle(idx)
    x = [e[idx] for e in x]
    y = [e[idx] for e in y]
    mask = [e[idx] for e in mask]
    return x, y, mask
    
# get memroy usage. x, y should be list
def memory_usage(x, y):
    total = 0
    total += sum([sys.getsizeof(e) for e in x])
    total += sum([sys.getsizeof(e) for e in y])
    return total
    
class Timer(object):
    def __init__(self):
        self.bgn_time = time.time()
        
    def show(self, string):
        now_time = time.time()
        f = "{0:<30} {1:<10}"
        time_diff = now_time - self.bgn_time
        print f.format(string, str(time_diff)+"s")