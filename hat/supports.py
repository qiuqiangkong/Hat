'''
SUMMARY:  support functions
AUTHOR:   Qiuqiang Kong
Created:  2016.05.17
Modified: -
--------------------------------------
'''
import numpy as np
import sys
import backend as K

### basic
# data to tuple
def to_tuple( x ):
    if type(x) is not tuple:
        x = (x,)
    return x
    
# data to list
def to_list( x ):
    if type(x) is not list:
        x = [x]
    return x
    
def is_one_element( x ):
    if len(x)==1: return True
    else: return False
    
def is_elem_equal( x ):
    return len( set( x ) ) <= 1
    
# Get mask from data (for RNN). size(mask): batch_num*n_time
def get_mask( input ):
    return K.neq( K.sum( K.abs( input ), axis=-1 ), 0. )
    
### data related
# shuffle data
def shuffle( xs, ys ):
    N = len( xs[0] )
    idx = np.arange(N)
    np.random.shuffle( idx )
    xs = [ x[idx] for x in xs ]
    ys = [ y[idx] for y in ys ]
    return xs, ys
    
# x, y, mask should be list of ndarray of same batch_num
def shuffle_xymask( x, y, mask ):
    N = len( x[0] )
    idx = np.arange(N)
    np.random.shuffle( idx )
    x = [ e[idx] for e in x ]
    y = [ e[idx] for e in y ]
    mask = [ e[idx] for e in mask ]
    return x, y, mask
    
# get memroy usage. x, y should be list
def memory_usage( x, y ):
    total = 0
    total += sum( [ sys.getsizeof(e) for e in x ] )
    total += sum( [ sys.getsizeof(e) for e in y ] )
    return total
    
### directed graph releated
# sub breadth first traversal
def BFT_sub( curr_layer, id_list, layer_list ):
    need_visits = []
    for layer in curr_layer.nexts_:
        if ( layer.id_ not in id_list ) and ( layer is not [] ):
            need_visits.append( layer )
            id_list.append( layer.id_ )
            layer_list.append( layer )    
            
    for layer in need_visits:
        id_list, layer_list = BFT_sub( layer, id_list, layer_list )
    
    return id_list, layer_list
    
# breadth first traversal
def BFT( in_layers ):
    in_layers = to_list( in_layers )
    id_list = []
    layer_list = []
    need_visits = []    # list of layer
    
    for layer in in_layers:
        id_list.append( layer.id_ )
        layer_list.append( layer )
        
        for next_layer in layer.nexts_:
            if ( next_layer not in need_visits ) and ( next_layer is not [] ):
                need_visits.append( next_layer )
                id_list.append( next_layer.id_ )
                layer_list.append( next_layer )

    # sub BFT
    for layer in need_visits:
        id_list, layer_list = BFT_sub( layer, id_list, layer_list )
    return id_list, layer_list