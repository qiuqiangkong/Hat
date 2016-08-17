'''
SUMMARY:  basic layers for building deep neural network
AUTHOR:   Qiuqiang Kong
Created:  2016.05.18
Modified: 2016.06.09 Add n_dim param to Flatten
          2016.08.02 Add info_(), load_from_info()
--------------------------------------
'''
import numpy as np
import inspect
from ..import backend as K
from ..import initializations
from ..import activations
from ..import regularizations 
from ..globals import new_id
from ..supports import to_tuple, to_list, is_one_element, is_elem_equal
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
    
    # -------- Public attributes --------
    
    # layer's id
    @property
    def id_( self ):
        return self._id_
        
    # layer's name
    @property
    def name_( self ):
        return self._name_

    # next layers
    @property
    def nexts_( self ):
        return self._nexts_
            
    # prev layers
    @property
    def prevs_( self ):
        return self._prevs_
           
    # output nodes
    @property
    def output_( self ):
        return self._output_
            
    # output shape
    @property
    def out_shape_( self ):
        return self._out_shape_
            
    # params (graph representation)
    @property
    def params_( self ):
        return self._params_
            
    # regularization value (graph representation)
    @property
    def reg_value_( self ):
        return self._reg_value_
    
    # -------- Public methods --------
    
    # add layer to this layer's nexts pointer
    def add_next( self, next_layer ):
        self._nexts_.append( next_layer )

    # -------- Private methods --------
    
    # Any class inherited this class should implement these attributes
    def _check_attributes( self ):
        attributes = [ '_id_', '_name_', '_prevs_', '_nexts_', '_output_', '_out_shape_', '_params_', '_reg_value_' ]
        for att in attributes:
            if hasattr( self, att ) is False:
                raise Exception( 'attribute ' + att + ' need to be inplemented!' )
                
    # Assign init value to weights. If init value is not given, then random is used. 
    def _init_params( self, init_val, init_type, shape, name ):
        if init_val is None:
            return K.shared( initializations.get( init_type )( shape ), name )
        else:
            return K.shared( init_val, name )

    # -------- Abstract attributes --------
    
    # layer's info & params (list of ndarray)
    @abstractproperty
    def info_( self ):
        pass
        
    # -------- Abstract methods --------
    
    # load layer from info
    @abstractmethod
    def load_from_info( self ):
        pass

    # ----------------------------------
    
    
    
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
        inputs = [ in_layer.output_ for in_layer in in_layers ]
        in_shapes = [ in_layer.out_shape_ for in_layer in in_layers ]
        
        # non_var_args will choose which _fn to use
        non_var_args = inspect.getargspec( self._fn )[0]       # num of arguments, not include **kwargs

        # case 1: num_input_layer=1
        if len( in_layers ) == 1:
            assert 'input' in non_var_args, "Your Lamda function do not have 'input' argument! "
            # overload 1, with in_shape argument
            if 'in_shape' in non_var_args:
                return_tuple = self._fn( inputs[0], in_shape=in_shapes[0], **self._kwargs_ )
                assert len(to_tuple(return_tuple))==2, "Your must return output, out_shape in your Lambda function!"
                (output, out_shape) = return_tuple
                assert type(out_shape) is tuple, "Check your out_shape in your Lambda function!"
            # overload 2, w/o in_shape argument
            else:
                 output = self._fn( inputs[0], **self._kwargs_ )
                 out_shape = in_shapes[0]
                 
        # case 2: num_input_layer>1
        elif len( in_layers ) > 1:
            assert 'inputs' in non_var_args, "Your Lamda function do not have 'inputs' argument! "
            # overload 1, with in_shape argument
            if 'in_shapes' in non_var_args:
                return_tuple = self._fn( inputs, in_shapes=in_shapes, **self._kwargs_ )
                assert len(to_tuple(return_tuple))==2, "Your must return output, out_shape in your Lambda function!"
                (output, out_shape) = return_tuple
                assert type(out_shape) is tuple, "Check your out_shape in your Lambda function!"
            # overload 2, w/o in_shape argument
            else:
                output = self._fn( inputs, **self._kwargs_ )
                assert is_elem_equal( in_shapes ), "Your Input Layers' shapes are not same. Check your model! " \
                        + "Or try add 'in_shapes' arguments and 'out_shape' to your Lambda function!"
                out_shape = in_shapes[0]
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = out_shape
        self._output_ = output
        self._params_ = []
        self._reg_value_ = 0.
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                             # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
    
    # layer's info & params
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'name': self._name_,
                 'id': self._id_, 
                 'fn': self._fn, 
                 'kwargs': self._kwargs_, }
        return dict
        
    # ---------- Public methods -------------
        
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        layer = cls( info['fn'], info['name'], **info['kwargs'] )
        return layer
        
    # ---------------------------------------


    
'''
Input Layer should be the first layer. 
'''
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
        self._reg_value_ = 0.
        
        # below are compulsory parts
        self._check_attributes()         # check if all attributes are implemented
        
    # ---------- Public attributes ----------
        
    # layer's info & params
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'name': self._name_, 
                 'in_shape': self._in_shape_, }
        return dict
        
    # ---------- Public methods ----------
        
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        layer = cls( in_shape=info['in_shape'], name=info['name'] )
        return layer
        
    # ------------------------------------
  
  
        
'''
Dense Layer
'''
class Dense( Layer ):
    def __init__( self, n_out, act='linear', init_type='glorot_uniform', 
                  W_init=None, b_init=None, W_reg=None, b_reg=None, trainable=True, name=None ):
                      
        super( Dense, self ).__init__( name )
        self._n_out_ = n_out
        self._init_type_ = init_type
        self._act_ = act
        self._W_init_ = W_init
        self._b_init_ = b_init
        self._W_reg_ = W_reg
        self._b_reg_ = b_reg
        self._trainable_ = trainable
        
    def __call__( self, in_layers ):
        # merge
        in_layers = to_list( in_layers )
        input = K.concatenate( [ layer.output_ for layer in in_layers ], axis=-1 )
        #assert input.ndim==2, "Try add Flatten layer before Dense layer!"
        n_in = sum( [ layer.out_shape_[-1] for layer in in_layers ] )
        
        # init W
        if self._act_=='softmax':
            self._W_ = self._init_params( self._W_init_, 'zeros', shape=(n_in, self._n_out_), name=str(self._name_)+'_W' )
        else:
            self._W_ = self._init_params( self._W_init_, self._init_type_, shape=(n_in, self._n_out_), name=str(self._name_)+'_W' )

        # init b
        self._b_ = self._init_params( self._b_init_, 'zeros', shape=(self._n_out_,), name=str(self._id_)+'_b' )
            
        # output
        lin_out = K.dot( input, self._W_ ) + self._b_
        output = activations.get( self._act_ )( lin_out )
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = in_layers[0].out_shape_[0:-1] + (self._n_out_,)
        self._output_ = output
        self.set_trainable( self._trainable_ )
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                             # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
        
    @property
    def W_( self ):
        return K.get_value( self._W_ )
        
    @property
    def b_( self ):
        return K.get_value( self._b_ )
        
    # layer's info & params
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'name': self._name_, 
                 'n_out': self._n_out_, 
                 'act': self._act_, 
                 'init_type': self._init_type_,     # mute if W is None
                 'W': self.W_, 
                 'b': self.b_, 
                 'W_reg_info': regularizations.get_info( self._W_reg_ ),
                 'b_reg_info': regularizations.get_info( self._b_reg_ ), }
        return dict
        
    # ---------- Public methods ----------
    
    # set trainable True or False
    def set_trainable( self, val ):
        self._trainable_ = val
        
        if self._trainable_ is True:
            self._params_ = [ self._W_, self._b_ ]
            self._reg_value_ = self._get_reg()
        else:
            self._params_ = []
            self._reg_value_ = 0.
    
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        W_reg = regularizations.get_obj( info['W_reg_info'] )
        b_reg = regularizations.get_obj( info['b_reg_info'] )

        layer = cls( n_out=info['n_out'], act=info['act'], init_type=info['init_type'], 
                     W_init=info['W'], b_init=info['b'], W_reg=W_reg, b_reg=b_reg, name=info['name'] )
                     
        return layer
    
    # ---------- Private methods ----------
        
    # get regularization
    def _get_reg( self ):
        reg_value = 0. 
        
        if self._W_reg_ is not None:
            reg_value += self._W_reg_.get_reg( [self._W_] )
            
        if self._b_reg_ is not None:
            reg_value += self._b_reg_.get_reg( [self._b_] )
            
        return reg_value
        
    # merge outputs of many layers to one output
    def _merge( in_layers ):
        return K.concatenate( [ layer.output_ for layer in in_layers ] )
        
    # ------------------------------------
        
    
          
# todo
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
        self._check_attributes()                             # check if all attributes are implemented
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
        self._check_attributes()                             # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
        
    # layer's info
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'name': self._name_,
                 'ndim': self._ndim_, }
        return dict
           
    # ---------- Public methods ----------
           
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        layer = cls( ndim=info['ndim'], name=info['name'] )
        return layer

    # ------------------------------------



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
        output = K.ifelse( K.eq( self._tr_phase_node_, 1. ), self._drop_out( input, self._p_drop_ ), input )
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = in_layer.out_shape_
        self._output_ = output
        self._params_ = []
        self._reg_value_ = 0.
        
        # below are compulsory parts
        [ layer.add_next(self) for layer in in_layers ]     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                             # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
        
    # layer's info
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'name': self._name_,
                 'p_drop': self._p_drop_, }
        return dict
        
    # ---------- Public methods ----------
    
    # load layer from info
    @classmethod
    def load_from_info( cls, info ):
        layer = cls( p_drop=info['p_drop'], name=info['name'] )
        return layer
    
    # ---------- Private methods ----------
    
    def _drop_out( self, input, p_drop ):
        if p_drop < 0. or p_drop >= 1:
            raise Exception('Dropout level must be in interval (0,1)')
        keep = K.rng_binomial( input.shape, 1.-p_drop )
        output = input * keep
        output /= (1.-p_drop)
        return output
    
    # ------------------------------------
        
    
        
    
           
    
    

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