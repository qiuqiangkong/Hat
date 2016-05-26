import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# from theano.ifelse import ifelse
import theano.ifelse

_FLOATX = theano.config.floatX

# Create a node of the graph. shape should be tuple
def placeholder( n_dim=None, name=None ):
    #return T.TensorType( dtype=_FLOATX, broadcastable=[False]*n_dim, name=name )()
    if n_dim==0: return T.scalar( name=name, dtype=_FLOATX )
    if n_dim==1: return T.vector( name=name, dtype=_FLOATX )
    if n_dim==2: return T.matrix( name=name, dtype=_FLOATX )
    if n_dim==3: return T.tensor3( name=name, dtype=_FLOATX )
    if n_dim==4: return T.tensor4( name=name, dtype=_FLOATX )
    
# shared tensor from numpy array
def sh_variable( value, name=None, dtype=_FLOATX ):
    return theano.shared( value=value.astype(dtype), name=name )

# format data    
def format_data( value, dtype=_FLOATX ):
    return value.astype( dtype )
    
# return value from GPU to CPU
def get_value( x ):
    return x.get_value()
    
### basic operation
   
def zeros( shape ):
    return T.zeros( shape, dtype=_FLOATX )
    
def zeros_like( x ):
    return T.zeros_like( x )
   
def sum( x, axis=None ):
    return T.sum( x, axis )
    
def log( x ):
    return T.log( x )
    
def exp( x ):
    return T.exp( x )

def mean( x, axis=None ):
    return T.mean( x, axis )
    
def std( x, axis=None ):
    return T.std( x, axis )

def dot( X, Y ):
    return T.dot( X, Y )
    
def sqr( x ):
    return T.sqr( x )
    
def sqrt( x ):
    return T.sqrt( x )
    
def abs( x ):
    return T.abs_( x )
    
def concatenate( inputs, axis ):
    return T.concatenate( inputs, axis )
    
def ndim( x ):
    return x.ndim
    
def max( x, axis=None ):
    return T.max( x, axis )
    
def clip( x, min_, max_ ):
    return T.clip( x, min_, max_ )
    
def eq( a, b ):
    return T.eq( a, b )
    
def neq( a, b ):
    return T.neq( a, b )
    
def gt( a, b ):
    return T.gt( a, b )
    
def lt( a, b ):
    return T.lt( a, b )
    
def ifelse( condition, op1, op2 ):
    return theano.ifelse.ifelse( condition, op1, op2 )
    
### activations
def softmax( x ):
    return T.nnet.softmax( x )
    
def tanh( x ):
    return T.tanh( x )
    
def sigmoid( x ):
    return T.nnet.sigmoid( x )
    
def relu( x, alpha, max_value ):
    y = T.nnet.relu( x, alpha )
    if max_value is not None:
        y = T.minimum( y, max_value )
    y += 1e-8       # used for avoiding masking
    return y
    
### objectives
# cross entropy loss. y_pred, y_gt should be 2D, DO NOT USE tensor.nnet.categorical_crossentropy
def categorical_crossentropy( p_y_pred, y_gt ):
    return T.mean( T.nnet.categorical_crossentropy( p_y_pred, y_gt ) )
    # DO NOT DELETE! This is manually written cross entropy. Some times no bug than T.nnet.categorical_crossentropy
    #z = T.argmax(y_gt, axis=-1)
    #return -T.mean( T.log(p_y_pred)[T.arange(z.shape[0]), z] )
    
def binary_crossentropy( p_y_pred, y_gt ):
    return T.mean( T.nnet.binary_crossentropy( p_y_pred, y_gt ) )
    

### training phase node
# A node to represent training 1.0 or testing 0.
tr_phase_node = placeholder( n_dim=0, name='tr_phase_node' )
    
### functions
# theano function, without givens
# using method eg. f( x, 0. )
def function_no_given( input_nodes, output_nodes ):
    f = theano.function( input_nodes + [tr_phase_node], output_nodes, on_unused_input='ignore' )
    return f

# theano function, all inputs correspond to givens
# using method eg. f( 42, 1. )
def function_given( batch_size, input_nodes, output_nodes, given_nodes, updates=None ):
    assert len(input_nodes)==len(given_nodes), "Number of input and given must be same!"
    index_node = T.iscalar()
    f = theano.function( [index_node, tr_phase_node], output_nodes, givens={
                    input_node:given_node[(index_node)*batch_size:(index_node+1)*batch_size] \
                    for input_node,given_node in zip(input_nodes, given_nodes) }, updates=updates, 
                    on_unused_input='ignore' )
    return f
    
# gradient
def grad( cost, params ):
    return theano.grad( cost, params )
    
# scan, interface is same as theano

scan = theano.scan
    
### dropout
def dropout( x, p_drop ):
    if p_drop < 0. or p_drop >= 1:
        raise Exception('Dropout level must be in interval (0,1)')
    seed = np.random.randint(10e6)
    rng = RandomStreams(seed)
    p_retain = 1. - p_drop
    xtr = x * rng.binomial(x.shape, p=p_retain, dtype=x.dtype)
    xtr /= p_retain
    xte = x
    z = theano.ifelse.ifelse( T.eq( tr_phase_node, 1. ), xtr, xte )
    return z