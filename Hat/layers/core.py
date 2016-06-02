'''
SUMMARY:  basic layers for building deep neural network
AUTHOR:   Qiuqiang Kong
Created:  2016.05.18
Modified: -
--------------------------------------
'''
import numpy as np
from ..import backend as K
from ..import initializations
from ..import activations
from ..globals import new_id
from ..supports import to_tuple, to_list, is_one_element
from abc import ABCMeta, abstractmethod, abstractproperty

'''
Layer is the base class of all Layer Classes
'''
class Layer( object ):
    __metaclass__ = ABCMeta
    
    def __init__( self, name ):
        self._id = new_id()
        if name is None:
            self._name = self.__class__.__name__ + '_' + str( self._id )
        else:
            self._name = name
    
    @property
    def id( self ):
        return self._id
        
    @property
    def name( self ):
        return self._name

    '''
    _nexts, _prevs, _output, _params, _regs are abstract properties. If you are implementing a new Class based on Layer, 
    the corresponding attributes of below must be implemented. 
    '''
    @property
    def nexts( self ):
        return self._nexts
            
    @property
    def prevs( self ):
        return self._prevs
            
    @property
    def output( self ):
        return self._output
            
    @property
    def out_shape( self ):
        return self._out_shape
            
    @property
    def params( self ):
        return self._params
            
    @property
    def reg_value( self ):
        return self._reg_value
            
    # Any class inherited this class should implement these attributes
    def check_attributes( self ):
        attributes = [ '_id', '_name', '_prevs', '_nexts', '_output', '_out_shape', '_params', '_reg_value' ]
        for att in attributes:
            if hasattr( self, att ) is False:
                raise Exception( 'attribute ' + att + ' need to be inplemented!' )
                
    # add layer to this layer's nexts pointer
    def add_next( self, next_layer ):
        self._nexts.append( next_layer )
        
'''
Lambda layer is a computation layer without params. 
You can define your own computation based on Lambda layer. 
'''
class Lambda( Layer ):
    def __init__( self, fn, name=None, **kwargs ):
        super( Lambda, self ).__init__( name )
        self._fn = fn
        self._kwargs = kwargs
        
    def __call__( self, in_layers ):
        in_layers = to_list( in_layers )
        
        # do lamda function
        inputs = [ in_layer.output for in_layer in in_layers ]
        in_shapes = [ in_layer.out_shape for in_layer in in_layers ]
        in_list = inputs + in_shapes
        output, out_shape = self._fn( *in_list, **self._kwargs )
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = out_shape
        self._output = output
        self._params = []
        self._reg_value = 0.
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
        
        
class InputLayer( Layer ):
    def __init__( self, in_shape, name=None ):
        super( InputLayer, self ).__init__( name )
        in_shape = to_tuple( in_shape )
        out_shape = (None,) + in_shape
        
        # assign attributes
        self._prevs = []
        self._nexts = []
        self._output = K.placeholder( n_dim=len(out_shape), name=self._name+'_out' )
        self._out_shape = out_shape
        self._params = []
        self._reg_value = 0
        
        # below are compulsory parts
        self.check_attributes()         # check if all attributes are implemented
        
        
'''
Dense Layer
'''
class Dense( Layer ):
    def __init__( self, n_out, act='linear', init_type='glorot_uniform', reg=None, W_init=None, b_init=None, name=None ):
        super( Dense, self ).__init__( name )
        self._n_out = n_out
        self._init_type = init_type
        self._act = act
        self._reg = reg
        self._W_init = W_init
        self._b_init = b_init
        
    def __call__( self, in_layers ):
        # merge
        in_layers = to_list( in_layers )
        input = K.concatenate( [ layer.output for layer in in_layers ], axis=-1 )
        #assert input.ndim==2, "Try add Flatten layer before Dense layer!"
        n_in = sum( [ layer.out_shape[-1] for layer in in_layers ] )
        
        # init W
        if self._W_init is None:
            if self._act=='softmax':
                self._W = initializations.get( 'zeros' )( (n_in, self._n_out), name=str(self._name)+'_W' )
            else:
                self._W = initializations.get( self._init_type )( (n_in, self._n_out), name=str(self._name)+'_W' )
        else: 
            self._W = K.sh_variable( self._W_init )
            
        # init b
        if self._b_init is None:
            self._b = initializations.get( 'zeros' )( self._n_out, name=str(self._id)+'_b' )
        else:
            self._b = K.sh_variable( self._b_init, name=str(self._id)+'_b' )
            
        # output
        lin_out = K.dot( input, self._W ) + self._b
        output = activations.get( self._act )( lin_out )
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        #self._out_shape = (None, self._n_out)
        self._out_shape = in_layers[0].out_shape[0:-1] + (self._n_out,)
        self._output = output
        self._params = [ self._W, self._b ]
        self._reg_value = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    # get regularization
    def _get_reg( self ):
        if self._reg is None:
            _reg_value = 0.
        else:
            _reg_value = self._reg.reg_value( [ self.W ] )
        return _reg_value
        
    # merge outputs of many layers to one output
    def _merge( in_layers ):
        return K.concatenate( [ layer.output for layer in in_layers ] )
        
    @property
    def W( self ):
        return self._W
        
    @property
    def b( self ):
        return self._b
           
'''
Merge Layer, can merge multi layers to one layer
Currently only support 2 dim merge
'''
class Merge( Layer ):
    def __init__( self, name=None ):
        super( Merge, self ).__init__( name )
        
    def __call__( self, in_layers ):
        in_layers = to_list( in_layers )
        assert self._check_dim( in_layers, n_dim=2 ), "The layers be merged must be n_dim=2 !"
        
        # concatenate inputs of layers to one output
        output = K.concatenate( [ layer.output for layer in in_layers ], axis=1 )
        new_dim = sum( [ layer.out_shape[1] for layer in in_layers ] )
        out_shape = (None, new_dim)
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = out_shape
        self._output = output
        self._params = []
        self._reg_value = 0.
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    # check if input dims are legal
    def _check_dim( self, in_layers, n_dim ):
        for layer in in_layers:
            if len( layer.out_shape ) != n_dim:
                return False
        return True

    
    
'''
Flatten Layer. Flatten mutli dim input to 2D output
'''
def _flatten( input, in_shape ):
    output = input.flatten(2)
    out_shape = ( None, np.prod( in_shape[1:] ) )
    return output, out_shape
    
class Flatten( Lambda ):
    def __init__( self, name=None ):
        super( Flatten, self ).__init__( _flatten, name )
    

'''
Dropout layer
'''
class Dropout( Layer ):
    def __init__( self, p_drop=0.1, name=None ):
        super( Dropout, self ).__init__( name )
        self._p_drop = p_drop
        self._tr_phase_node = K.common_tr_phase_node
        
    def __call__( self, in_layers ):
        # only one input layer is allowed
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Dense can only be one layer!"
        in_layer = in_layers[0]
        
        # dropout
        input = in_layer.output
        output = K.ifelse( K.eq( self._tr_phase_node, 1. ), self._tr_phase( input, self._p_drop ), input )
        
        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = in_layer.out_shape
        self._output = output
        self._params = []
        self._reg_value = 0.
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    def _tr_phase( self, input, p_drop ):
        if p_drop < 0. or p_drop >= 1:
            raise Exception('Dropout level must be in interval (0,1)')
        keep = K.rng_binomial( input.shape, 1.-p_drop )
        output = input * keep
        output /= (1.-p_drop)
        return output
    
    
'''
Batch normalization layer
'''
class BN( Layer ):
    def __init__( self, name=None ):
        super( BN, self ).__init__( name )
        self._epsilon = 1e-8
        self._momentum = 0.9
        self._global_mean = 0.
        self._global_var = 0.
        self._tr_phase_node = K.tr_phase_node
        
    def __call__( self, in_layers ):
        # only one input layer is allowed
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Dense can only be one layer!"
        in_layer = in_layers[0]
        
        # init params
        assert len( in_layer.out_shape )==2, "Only support 2 dim input batch normalization!"
        input = in_layer.output
        n_in = in_layer.out_shape[1]
        self._gamma = initializations.get( 'ones' )( n_in, name=self._name+'_gamma' )
        self._beta = initializations.get( 'zeros' )( n_in, name=self._name+'_beta' )
        
        # do batch normalization
        output = K.ifelse( K.eq( self._tr_phase_node, 1. ), self._tr_phase(input), self._te_phase(input) )

        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = in_layer.out_shape
        self._output = output
        self._params = [ self._gamma, self._beta ]
        self._reg_value = 0.
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    def _tr_phase( self, input ):
        mean_ = K.mean( input, axis=0 )             # size: n_in
        var_ = K.sqr( K.std( input, axis=0 ) )      # size: n_in
        x_hat = input / K.sqrt( var_ + self._epsilon )
        output = x_hat * self._gamma + self._beta
        
        # update global mean & var
        # this implementation ignore the m/(m-1) of var where m is batch_size compared with original paper
        self._global_mean = self._momentum * self._global_mean + ( 1 - self._momentum ) * mean_
        self._global_var = self._momentum * self._global_var + ( 1 - self._momentum ) * var_
        
        return output
        
    def _te_phase( self, input ):
        output = ( input - self._global_mean ) / K.sqrt( self._global_var + self._epsilon ) * self._gamma + self._beta
        return output
        
    @property
    def gamma( self ):
        return self._gamma
        
    @property
    def beta( self ):
        return self._beta