'''
SUMMARY:  basic layers for building deep neural network
AUTHOR:   Qiuqiang Kong
Created:  2016.05.18
Modified: 2016.06.09 Add n_dim param to Flatten
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
        self._id_ = new_id()
        if name is None:
            self._name_ = self.__class__.__name__ + '_' + str( self._id_ )
        else:
            self._name_ = name
    
    @property
    def id_( self ):
        return self._id_
        
    @property
    def name_( self ):
        return self._name_

    '''
    Abstract properties. including _nexts, _prevs, _output, _params, _regs are . If you are implementing a new Class based on Layer, 
    the corresponding attributes of below must be implemented. 
    '''
    @property
    def nexts_( self ):
        return self._nexts_
            
    @property
    def prevs_( self ):
        return self._prevs_
            
    @property
    def output_( self ):
        return self._output_
            
    @property
    def out_shape_( self ):
        return self._out_shape_
            
    @property
    def params_( self ):
        return self._params_
            
    @property
    def reg_value_( self ):
        return self._reg_value_
            
    # Any class inherited this class should implement these attributes
    def check_attributes( self ):
        attributes = [ '_id_', '_name_', '_prevs_', '_nexts_', '_output_', '_out_shape_', '_params_', '_reg_value_' ]
        for att in attributes:
            if hasattr( self, att ) is False:
                raise Exception( 'attribute ' + att + ' need to be inplemented!' )
                
    # add layer to this layer's nexts pointer
    def add_next( self, next_layer ):
        self._nexts_.append( next_layer )
        
    '''
    abstract methods
    '''
    '''
    # serialize layer
    @abstractmethod
    def serialize( self ):
        pass
    '''
        
'''
Lambda layer is a computation layer without params. 
You can define your own computation based on Lambda layer. 
'''
class Lambda( Layer ):
    def __init__( self, fn, name=None, **kwargs ):
        super( Lambda, self ).__init__( name )
        self._fn = fn
        self._kwargs_ = kwargs
        
    def __call__( self, in_layers ):
        in_layers = to_list( in_layers )
        
        # do lamda function
        inputs = [ in_layer.output_ for in_layer in in_layers ]
        in_shapes = [ in_layer.out_shape_ for in_layer in in_layers ]
        in_list = inputs + in_shapes
        output, out_shape = self._fn( *in_list, **self._kwargs_ )
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = out_shape
        self._output_ = output
        self._params_ = []
        self._reg_value_ = 0.
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    
        
class InputLayer( Layer ):
    def __init__( self, in_shape, name=None ):
        super( InputLayer, self ).__init__( name )
        self._in_shape_ = to_tuple( in_shape )
        out_shape = (None,) + self._in_shape_
        
        # assign attributes
        self._prevs_ = []
        self._nexts_ = []
        self._output_ = K.placeholder( n_dim=len(out_shape), name=self._name_+'_out' )
        self._out_shape_ = out_shape
        self._params_ = []
        self._reg_value_ = 0
        
        # below are compulsory parts
        self.check_attributes()         # check if all attributes are implemented
        
    # model's info & params
    @property
    def info_( self ):
        dict = { 'id': self._id_, 
                 'name': self._name_, 
                 'in_shape': self._in_shape_, }
        return dict
        
    @classmethod
    def load_from_info( cls, info ):
        layer = cls( in_shape=info['in_shape'], name=info['name'] )
        return layer
'''
Dense Layer
'''
class Dense( Layer ):
    def __init__( self, n_out, act='linear', init_type='glorot_uniform', reg=None, W_init=None, b_init=None, name=None ):
        super( Dense, self ).__init__( name )
        self._n_out_ = n_out
        self._init_type_ = init_type
        self._act_ = act
        self._reg_ = reg
        self._W_init_ = W_init
        self._b_init_ = b_init
        
    def __call__( self, in_layers ):
        # merge
        in_layers = to_list( in_layers )
        input = K.concatenate( [ layer.output_ for layer in in_layers ], axis=-1 )
        #assert input.ndim==2, "Try add Flatten layer before Dense layer!"
        n_in = sum( [ layer.out_shape_[-1] for layer in in_layers ] )
        
        # init W
        if self._W_init_ is None:
            if self._act_=='softmax':
                self._W_ = initializations.get( 'zeros' )( (n_in, self._n_out_), name=str(self._name_)+'_W' )
            else:
                self._W_ = initializations.get( self._init_type_ )( (n_in, self._n_out_), name=str(self._name_)+'_W' )
        else: 
            self._W_ = K.sh_variable( self._W_init_ )
            
        # init b
        if self._b_init_ is None:
            self._b_ = initializations.get( 'zeros' )( self._n_out_, name=str(self._id_)+'_b' )
        else:
            self._b_ = K.sh_variable( self._b_init_, name=str(self._id_)+'_b' )
            
        # output
        lin_out = K.dot( input, self._W_ ) + self._b_
        output = activations.get( self._act_ )( lin_out )
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        #self._out_shape = (None, self._n_out)
        self._out_shape_ = in_layers[0].out_shape_[0:-1] + (self._n_out_,)
        self._output_ = output
        self._params_ = [ self._W_, self._b_ ]
        self._reg_value_ = self._get_reg()
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    # get regularization
    def _get_reg( self ):
        if self._reg_ is None:
            reg_value = 0.
        else:
            reg_value = self._reg_value_( [ self.W ] )
        return reg_value
        
    # merge outputs of many layers to one output
    def _merge( in_layers ):
        return K.concatenate( [ layer.output_ for layer in in_layers ] )
        
    @property
    def W_( self ):
        return K.get_value( self._W_ )
        
    @property
    def b_( self ):
        return K.get_value( self._b_ )
        
    # model's info & params
    @property
    def info_( self ):
        dict = { 'id': self._id_, 
                 'name': self._name_, 
                 'n_out': self._n_out_, 
                 'act': self._act_, 
                 'init_type': self._init_type_,     # mute if W is None
                 'reg': self._reg_, 
                 'W': self.W_, 
                 'b': self.b_, }
        return dict
           
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        layer = cls( n_out=info['n_out'], act=info['act'], init_type=info['init_type'], 
                     reg=info['reg'], W_init=info['W'], b_init=info['b'], name=info['name'] )
        return layer
           
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
        output = K.concatenate( [ layer.output_ for layer in in_layers ], axis=1 )
        new_dim = sum( [ layer.out_shape_[1] for layer in in_layers ] )
        out_shape = (None, new_dim)
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = out_shape
        self._output_ = output
        self._params_ = []
        self._reg_value_ = 0.
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    # check if input dims are legal
    def _check_dim( self, in_layers, n_dim ):
        for layer in in_layers:
            if len( layer.out_shape_ ) != n_dim:
                return False
        return True
        
    # layer's info & params
    @property
    def info_( self ):
        dict = { 'id': self._id_, 
                 'name': self._name_ }
        return dict
           
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        layer = cls( name=info['name'] )
        return layer

    
    
'''
Flatten Layer. Flatten mutli dim input to 2D output
'''
'''
def _flatten( input, in_shape ):
    output = input.flatten(2)
    out_shape = ( None, np.prod( in_shape[1:] ) )
    return output, out_shape
    
class Flatten( Lambda ):
    def __init__( self, name=None ):
        super( Flatten, self ).__init__( _flatten, name )
'''
class Flatten( Layer ):
    def __init__( self, ndim=2, name=None ):
        super( Flatten, self ).__init__( name )
        self._ndim_ = ndim
        
    def __call__( self, in_layers ):
        # only one input layer is allowed
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Dense can only be one layer!"
        in_layer = in_layers[0]
        
        # do flatten
        input = in_layer.output_
        in_shape = in_layer.out_shape_
        output = input.flatten( self._ndim_ )
        if self._ndim_==1:
            out_shape = in_shape[0]
        else:
            out_shape = in_shape[0:self._ndim_-1] + ( np.prod( in_shape[self._ndim_-1:] ), )
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = out_shape
        self._output_ = output
        self._params_ = []
        self._reg_value_ = 0.
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self.check_attributes()                             # check if all attributes are implemented
        return self
        
    # layer's info
    @property
    def info_( self ):
        dict = { 'id': self._id_, 
                 'ndim': self._ndim_, 
                 'name': self._name_ }
        return dict
           
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        layer = cls( ndim=info['ndim'], name=info['name'] )
        return layer

'''
Dropout layer
'''
class Dropout( Layer ):
    def __init__( self, p_drop=0.1, name=None ):
        super( Dropout, self ).__init__( name )
        self._p_drop_ = p_drop
        self._tr_phase_node_ = K.common_tr_phase_node
        
    def __call__( self, in_layers ):
        # only one input layer is allowed
        in_layers = to_list( in_layers )
        assert len(in_layers)==1, "The input of Dense can only be one layer!"
        in_layer = in_layers[0]
        
        # dropout
        input = in_layer.output_
        output = K.ifelse( K.eq( self._tr_phase_node_, 1. ), self._tr_phase( input, self._p_drop_ ), input )
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = in_layer.out_shape_
        self._output_ = output
        self._params_ = []
        self._reg_value_ = 0.
        
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
        
    # layer's info
    @property
    def info_( self ):
        dict = { 'id': self._id_, 
                 'p_drop': self._p_drop_, 
                 'name': self._name_ }
        return dict
           
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        layer = cls( p_drop=info['p_drop'], name=info['name'] )
        return layer
    

# todo
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
        assert len( in_layer.out_shape_ )==2, "Only support 2 dim input batch normalization!"
        input = in_layer.output
        n_in = in_layer.out_shape_[1]
        self._gamma = initializations.get( 'ones' )( n_in, name=self._name+'_gamma' )
        self._beta = initializations.get( 'zeros' )( n_in, name=self._name+'_beta' )
        
        # do batch normalization
        output = K.ifelse( K.eq( self._tr_phase_node, 1. ), self._tr_phase(input), self._te_phase(input) )

        # assign attributes
        self._prevs = in_layers
        self._nexts = []
        self._out_shape = in_layer.out_shape_
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