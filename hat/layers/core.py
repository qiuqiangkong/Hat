"""
SUMMARY:  basic layers for building deep neural network
AUTHOR:   Qiuqiang Kong
Created:  2016.05.18
Latest modified: 2017.02.10
--------------------------------------
"""
import numpy as np
import inspect
from ..import backend as K
from ..import initializations
from ..import activations
from ..import regularizations 
from ..globals import new_id
from ..supports import to_tuple, to_list, is_one_element, is_elem_equal
from abc import ABCMeta, abstractmethod, abstractproperty



class Layer(object):
    """The base class of all Layer Classes
    
    Args:
      kwargs: dictionary. 
    """
    __metaclass__ = ABCMeta
    
    def __init__(self , **kwargs):
        # self._check_kwargs_(kwargs)
        self._base_allowed_kwargs_ = ['id', 'name']
        
        if 'id' in kwargs:
            self._id_ = kwargs['id']
        else:
            self._id_ = new_id()
            
        if 'name' in kwargs:
            self._name_ = kwargs['name']
        else:
            self._name_ = self.__class__.__name__ + '_' + str(self._id_)
            
        self._base_kwargs_ = {'id': self._id_, 'name': self._name_}
            
        # should be assigned in __call__()
        self._prevs_ = None
        self._nexts_ = None
        self._out_shape_ = None
        self._params_ = None
        self._reg_value_ = None
        
        # should be assigend in compile()
        self._inputs_ = None
        self._output_ = None
    
    # -------- Public attributes --------
    
    # layer's id
    @property
    def id_(self):
        return self._id_
        
    # layer's name
    @property
    def name_(self):
        return self._name_

    # next layers
    @property
    def nexts_(self):
        return self._nexts_
            
    # prev layers
    @property
    def prevs_(self):
        return self._prevs_
           
    # Input nodes
    @property
    def inputs_(self):
        return self._inputs_
           
    # output nodes
    @property
    def output_(self):
        return self._output_
            
    # output shape
    @property
    def out_shape_(self):
        return self._out_shape_
            
    # params (graph representation)
    @property
    def params_(self):
        return self._params_
            
    # regularization value (graph representation)
    @property
    def reg_value_(self):
        return self._reg_value_
    
    # -------- Public methods --------
    
    def set_trainable_params_and_update_reg(self, trainable_params):
        self._params_ = []
        for ch in trainable_params:
            assert ch in self._legal_params_, "'ch' is not a param of " + self.__class__.__name__ + "! "
            self._params_.append(self.__dict__['_'+ch+'_'])
            
        self._reg_value_ = self._get_reg()
        
    
    # add layer to this layer's nexts pointer
    def add_next(self, next_layer):
        self._nexts_.append(next_layer)
        
    def set_previous(self, prevs_layer):
        self._prevs_ = [prevs_layer]
        
    def _check_kwargs(self, kwargs):
        for key in kwargs:
            if key not in self._base_allowed_kwargs_:
                raise Exception("'" + key + "' is not an allowed argument!")
        return
        
    # -------- Private methods --------
    
    # Add self to all prev layers' nexts pointer
    def _add_self_to_prevs_layer(self, in_layers):
        for layer in in_layers:
            layer.add_next(self)
    
    # Any class inherited this class should implement these attributes
    def _check_attributes(self):
        attributes =['_id_', '_name_', '_prevs_', '_nexts_', '_output_', '_out_shape_', '_params_', '_reg_value_']
        for att in attributes:
            if hasattr(self, att) is False:
                raise Exception('attribute ' + att + ' need to be inplemented!')
                
    # Assign init value to weights. If init value is not given, then random is used. 
    def _init_params(self, init_value, init_type, shape, name=None):
        if init_value is None:
            if type(init_type) is str:
                return K.shared(initializations.get(init_type)(shape), name)
            # if init_type is function
            else:
                return K.shared(init_type, name)
        else:
            return K.shared(init_value, name)
            
    # -------- Abstract attributes --------
    
    # layer's info & params (list of ndarray)
    @abstractproperty
    def info_(self):
        pass
        
    # -------- Abstract methods --------
    
    # load layer from info
    @abstractmethod
    def load_from_info(self):
        pass

    # ----------------------------------
    
    
class Template(Layer):
    """Template for users to create a Layer. 
    """
    def __init__(self, W_init, trainable_params=['W'], **kwargs):
        self._legal_params_ = ['W']
        super(Template, self).__init__(**kwargs)
        self._W_init_ = W_init
        self._trainable_params_ = trainable_params
        """
        # self._legal_params_ = []
        """
        
    def __call__(self, in_layers):
        """
        # self._prevs_
        # self._nexts_
        # self._out_shape_
        # init shared params
        # self._params_
        # self._reg_value_
        """
        pass
        
    def compile(self):
        """
        # self._inputs_
        # self._output_
        """
        pass
        
    def _get_reg(self):
        # return reg_value
        pass
        
    @property
    def info_(self):
        # return dict
        pass
        
    @classmethod
    def load_from_info(cls, info):
        # return layer
        pass
    

class InputLayer(Layer):
    """Input Layer should be the first layer of neural network. 
    
    Args:
      in_shape: tuple of integers. Input shape, e.g. (784,) or (3,28,28)
      kwargs: see Layer class for details. 
    """
    def __init__(self, in_shape, **kwargs):
        super(InputLayer, self).__init__(**kwargs)
        self._in_shape_ = to_tuple(in_shape)
        
        # Assign attributes
        self._params_ = []
        self._reg_value_ = 0.
        self._out_shape_ = (None,) + self._in_shape_
        self._prevs_ = []
        self._nexts_ = []
        
    # ---------- Public attributes ----------
        
    def compile(self):
        if self._prevs_:
            assert len(self._prevs_) == 1, "At most 1 layer can be fed to InputLayer!"
            input = self._prevs_[0].output_
        else:
            input = K.placeholder(n_dim=len(self._out_shape_), 
                                  name=self._name_+'_input'
                                  )
            
        self._inputs_ = [input]
        self._output_ = input
        
    @property
    def info_(self):
        kwargs = self._base_kwargs_
        info = {'class_name': self.__class__.__name__, 
                 'in_shape': self._in_shape_, 
                 'kwargs': kwargs}
        return info
        
    # ---------- Public methods ----------

    @classmethod
    def load_from_info(cls, info):
        layer = cls(in_shape=info['in_shape'], **info['kwargs'])
        return layer
        
    # ------------------------------------
 


class Dense(Layer):
    """Dense Layer. 
    
    Args:
      n_out: integar. Number of output units. 
      act: string | function. Non linear activation function. 
      init_type: string. See initializations.py for detials. 
      W_init: ndarray. 
      b_init: ndarray. 
      W_reg: regularization object. 
      b_reg: regularization object. 
      trainable_params: list of strings. Parameters be updated in training. 
      kwargs: see Layer class for details. 
    """
    def __init__(self, n_out, act, init_type='glorot_uniform', 
                  W_init=None, b_init=None, W_reg=None, b_reg=None, 
                  trainable_params=['W','b'], **kwargs):
        self._legal_params_ = ['W', 'b']
                      
        super(Dense, self).__init__(**kwargs)   # assign id, name
        self._n_out_ = n_out
        self._init_type_ = init_type
        self._act_ = act
        self._W_init_ = W_init
        self._b_init_ = b_init
        self._W_reg_ = W_reg
        self._b_reg_ = b_reg
        self._trainable_params_ = trainable_params
         
    def __call__(self, in_layers):
        # todo, check in_layers shapes equal
        in_layers = to_list(in_layers)
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = in_layers[0].out_shape_[0:-1] + (self._n_out_,)
        
        # shared params
        n_in = np.sum([layer.out_shape_[-1] for layer in in_layers])
        if not hasattr(self, '_W_'):
            self._W_ = self._init_params(self._W_init_, self._init_type_, 
                                        shape=(n_in, self._n_out_), 
                                        name=str(self._name_)+'_W'
                                        )
        if not hasattr(self, '_b_'):
            self._b_ = self._init_params(self._b_init_, 'zeros', 
                                        shape=(self._n_out_,), 
                                        name=str(self._name_)+'_b'
                                        )
        
        # set params & update reg_value
        self.set_trainable_params_and_update_reg(self._trainable_params_)
        
        # below are compulsory parts
        self._add_self_to_prevs_layer(in_layers)     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                       # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
        
    @property
    def W_(self):
        return K.get_value(self._W_)
        
    @property
    def b_(self):
        return K.get_value(self._b_)

    @property
    def info_(self):
        kwargs = self._base_kwargs_
        info = {'class_name': self.__class__.__name__,
                 'n_out': self._n_out_, 
                 'act': self._act_, 
                 'init_type': self._init_type_,     # mute if W is None
                 'W': self.W_, 
                 'b': self.b_, 
                 'W_reg_info': regularizations.get_info(self._W_reg_),
                 'b_reg_info': regularizations.get_info(self._b_reg_), 
                 'trainable_params': self._trainable_params_, 
                 'kwargs': kwargs}
        return info
        
    # ---------- Public methods ----------
    
    def compile(self):
        self._inputs_ = [layer.output_ for layer in self._prevs_]
        input = K.concatenate(self._inputs_, axis=-1)
        self._output_ = self._get_output(input)

    @classmethod
    def load_from_info(cls, info):
        W_reg = regularizations.get_obj(info['W_reg_info'])
        b_reg = regularizations.get_obj(info['b_reg_info'])

        layer = cls(n_out=info['n_out'], act=info['act'], init_type=info['init_type'], 
                     W_init=info['W'], b_init=info['b'], W_reg=W_reg, b_reg=b_reg, 
                     trainable_params=info['trainable_params'], **info['kwargs'])
                     
        return layer

    # ---------- Private methods ----------

    def _get_reg(self):
        reg_value = 0. 
        
        if self._W_reg_ is not None:
            if 'W' in self._trainable_params_:
                reg_value += self._W_reg_.get_reg([self._W_])
        if self._b_reg_ is not None:
            if 'b' in self._trainable_params_:
                reg_value += self._b_reg_.get_reg([self._b_])

        return reg_value
        
    def _get_output(self, input):
        lin_out = K.dot(input, self._W_) + self._b_
        if type(self._act_) is str:
            output = activations.get(self._act_)(lin_out)
        else:
            output = self._act_(lin_out)
        return output
 
    # ------------------------------------
    
    
class Activation(Layer):
    """Nonlinear layer. 
    
    Args:
      act_func: string | function. Nonlinearity mapping. 
      kwargs: see Layer class for details. 
    """
    def __init__(self, act_func, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self._act_func = act_func
        
    def __call__(self, in_layers):
        # only one input layer is allowed
        in_layers = to_list(in_layers)
        assert len(in_layers)==1, "The input of Activation can only be one layer!"
        in_layer = in_layers[0]

        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = in_layer.out_shape_
        self._params_ = []
        self._reg_value_ = 0.
        
        # below are compulsory parts
        self._add_self_to_prevs_layer(in_layers)           # add this layer to all prev layers' nexts pointer
        self._check_attributes()                             # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------

    @property
    def info_(self):
        kwargs = self._base_kwargs_
        info = {'class_name': self.__class__.__name__, 
                 'act_func': self._act_func, 
                 'kwargs': kwargs}
        return info
           
    # ---------- Public methods ----------
           
    def compile(self):
        self._inputs_ =[layer.output_ for layer in self._prevs_]
        input = self._inputs_[0]
        self._output_ = self._get_output(input)

    @classmethod
    def load_from_info(cls, info):
        layer = cls(act_func=info['act_func'], **info['kwargs'])
        return layer

    # ---------- Private methods ----------
    
    def _get_output(self, input):
        if type(self._act_func) is str:
            return activations.get(self._act_func)(input)
        if callable(self._act_func):
            return self._act_func(input)
        
    
class Lambda(Layer):
    """
    Lambda layer is a user self defined computation layer without params. 
    """
    def __init__(self, fn, **kwargs):
        self._fn = fn
        self._kwargs_ = kwargs
        super(Lambda, self).__init__(**kwargs)
        
    def __call__(self, in_layers):
        in_layers = to_list(in_layers)
        self._prevs_ = in_layers
        self._nexts_ = []
        in_shapes =[layer.out_shape_ for layer in in_layers]
        self._out_shape_ = self._calculate_out_shape(in_shapes)
        self._params_ = []
        self._reg_value_ = 0.
        
        # below are compulsory parts
        self._add_self_to_prevs_layer(in_layers)     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                       # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
    
    def compile(self):
        self._inputs_ =[layer.output_ for layer in self._prevs_]
        self._output_ = self._get_output(self._inputs_)

    @property
    def info_(self):
        kwargs = self._kwargs_
        kwargs.update(self._base_kwargs_)
        info = {'class_name': self.__class__.__name__, 
                 'fn': self._fn, 
                 'kwargs': kwargs}
        return info
        
    # ---------- Public methods -------------

    @classmethod
    def load_from_info(cls, info):
        layer = cls(info['fn'], **info['kwargs'])
        return layer
        
    # ---------- Private methods ------------
    
    def _get_pseudo_test_num(self):
        if 'pseudo_test_num' in self._kwargs_.keys():
            return self._kwargs_['pseudo_test_num']
        else:
            return 1
    
    def _calculate_out_shape(self, in_shapes):
        """use pseudo data to get out_shape
        """
        us = []
        dict = {}
        for in_shape in in_shapes:
            u = K.placeholder(len(in_shape))
            pseudo_test_num = self._get_pseudo_test_num()
            data = K.format_data(np.zeros((pseudo_test_num,)+in_shape[1:]))   # pseudo data
            us.append(u)
            dict[u] = data
        
        if len(in_shapes)==1:
            shape = self._fn(us[0], **self._kwargs_).eval(dict).shape
        else:
            shape = self._fn(us, **self._kwargs_).eval(dict).shape
            
        return (None,) + shape[1:]
        
    def _get_output(self, inputs):
        non_var_args = inspect.getargspec(self._fn)[0]       # num of arguments, not include **kwargs
        
        if len(self._prevs_) == 1:
            assert 'input' in non_var_args, "Your Lamda function do not have 'input' argument! "
            return self._fn(inputs[0], **self._kwargs_)
        elif len(self._prevs_) > 1:
            assert 'inputs' in non_var_args, "Your Lamda function do not have 'inputs' argument! "
            return self._fn(inputs, **self._kwargs_)

    # ---------------------------------------


def _flatten(input, **kwargs):
    ndim = kwargs['ndim']
    return input.flatten(ndim)
    
class Flatten(Lambda):
    """Flatten layer. 
    
    Args:
      ndim: integar. Dimension to flatten. 
    """
    def __init__(self, ndim=2, **kwargs):
        kwargs['ndim'] = ndim
        self._ndim_ = ndim
        super(Flatten, self).__init__(_flatten, **kwargs)
        
    # ---------- Public attributes ----------
    
    @property
    def info_(self):
        kwargs = self._base_kwargs_
        info = {'class_name': self.__class__.__name__, 
                 'ndim': self._ndim_, 
                 'kwargs': kwargs}
        return info
        
    # ---------- Public methods ----------
    
    @classmethod
    def load_from_info(cls, info):
        layer = cls(info['ndim'], **info['kwargs'])
        return layer
    
    # ------------------------------------


def _reshape(input, **kwargs):
    out_shape = kwargs['out_shape']
    n_samples = input.shape[0]
    # TODO 
    # assert np.prod(in_shape) == np.prod(out_shape), "Reshape class np.prod(in_shape) is not correct!"
    output = input.reshape((n_samples,) + out_shape)
    return output
    
class Reshape(Lambda):
    
    def __init__(self, out_shape, **kwargs):
        out_shape = to_tuple(out_shape)
        kwargs['out_shape'] = out_shape
        super(Reshape, self).__init__(_reshape, **kwargs)
        
    # ---------- Public attributes ----------
        
    @property
    def info_(self):
        kwargs = self._base_kwargs_
        dict = {'class_name': self.__class__.__name__, 
                'out_shape': self._out_shape_[1:],
                'kwargs': kwargs}
        return dict
           
    # ---------- Public methods ----------
           
    @classmethod
    def load_from_info(cls, info):
        layer = cls(info['out_shape'], **info['kwargs'])
        return layer

    # ------------------------------------
        
        

class Dropout(Layer):
    """Dropout layer. Ref: "Dropout: a simple way to prevent neural networks from overfitting."
    
    Args:
      p_drop: real value betwen [0,1].
      kwargs: see Layer class for details. 
    """
    def __init__(self, p_drop, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self._p_drop_ = p_drop
        self._tr_phase_node_ = K.common_tr_phase_node
        
    def __call__(self, in_layers):
        # only one input layer is allowed
        in_layers = to_list(in_layers)
        assert len(in_layers)==1, "The input of Dense can only be one layer!"
        in_layer = in_layers[0]
        
        # assign attributes
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = in_layer.out_shape_
        self._params_ = []
        self._reg_value_ = 0.
        
        # below are compulsory parts
        self._add_self_to_prevs_layer(in_layers)           # add this layer to all prev layers' nexts pointer
        self._check_attributes()                             # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
        
    @property
    def info_(self):
        kwargs = self._base_kwargs_
        info = {'class_name': self.__class__.__name__, 
                 'p_drop': self._p_drop_, 
                 'kwargs': kwargs}
        return info
        
    # ---------- Public methods ----------
    
    def compile(self):
        self._inputs_ =[layer.output_ for layer in self._prevs_]
        input = self._inputs_[0]
        self._output_ = self._get_output(input)
    
    @classmethod
    def load_from_info(cls, info):
        layer = cls(p_drop=info['p_drop'], **info['kwargs'])
        return layer
    
    # ---------- Private methods ----------
    
    def _drop_out(self, input, p_drop):
        if p_drop < 0. or p_drop >= 1:
            raise Exception('Dropout level must be in interval (0,1)')
        keep = K.rng_binomial(input.shape, 1.-p_drop)
        output = input * keep
        output /= (1.-p_drop)
        return output

    def _get_output(self, input):
        output = K.ifelse(K.eq(self._tr_phase_node_, 1.), self._drop_out(input, self._p_drop_), input)
        return output
    
    # ------------------------------------

    