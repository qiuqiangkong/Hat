import numpy as np
import inspect
from ..import backend as K
from ..import initializations
from ..import activations
from ..import regularizations 
from ..globals import new_id
from ..supports import to_tuple, to_list, is_one_element, is_elem_equal
from core import Layer


class BN(Layer):
    """ Batch normalization layer
    Ref: Ioffe, Sergey, et al. "Batch normalization: Accelerating deep network training by reducing internal covariate shift.", 2015
    
    Args
      axis: integar | tuple. Along which axes to normalize, e.g. axis=(0,) for DNN; axis=(0,2,3) for CNN. 
      gamma_init: real value (default 1.)
      beta_init: real value (default 0.)
      running_mean_init: real value (default 0.)
      running_var_init: real value (default 1.)
      trainable_params: list of strings. Allowed 'gamma', 'beta'
      kwargs: see Layer class for details. 
    """
    def __init__(self, axis, gamma_init=None, beta_init=None, running_mean_init=None, running_var_init=None, trainable_params=['gamma', 'beta'], **kwargs):
        self._legal_params_ = ['gamma', 'beta']
        super(BN, self).__init__(**kwargs)
        self._gamma_init_ = gamma_init
        self._beta_init_ = beta_init
        self._axis_ = to_tuple(axis)
        self._eps_ = 1e-6
        self._momentum_ = 0.99
        self._running_mean_init_ = running_mean_init
        self._running_var_init_ = running_var_init
        self._trainable_params_ = trainable_params
        self._tr_phase_node_ = K.common_tr_phase_node
        self._bn_param_shape_ = None
        self._inner_updates_ = None
        
    def __call__(self, in_layers):
        in_layers = to_list(in_layers)
        assert len(in_layers)==1, "The input can only be one layer!"
        
        self._prevs_ = in_layers
        self._nexts_ = []
        self._out_shape_ = in_layers[0].out_shape_
        
        # init params
        bn_param_shape = self._get_bn_param_shape(in_layers[0].out_shape_, self._axis_)
        self._bn_param_shape_ = bn_param_shape
        self._gamma_ = self._init_params(self._gamma_init_, 'ones', shape=bn_param_shape, 
                                          name=str(self._name_)+'_gamma')
        self._beta_ = self._init_params(self._beta_init_, 'zeros', shape=bn_param_shape, 
                                         name=str(self._name_)+'_beta')                                         
        self._running_mean_ = self._init_params(self._running_mean_init_, 'zeros', shape=bn_param_shape, 
                                         name=str(self._name_)+'_running_mean')
        self._running_var_ = self._init_params(self._running_var_init_, 'ones', shape=bn_param_shape, 
                                         name=str(self._name_)+'_running_var')                                
        
        # set params & update reg_value
        self.set_trainable_params_and_update_reg(self._trainable_params_)
        
        # below are compulsory parts
        self._add_self_to_prevs_layer(in_layers)     # add this layer to all prev layers' nexts pointer
        self._check_attributes()                       # check if all attributes are implemented
        return self
        
    # ---------- Public attributes ----------
    
    @property
    def gamma_(self):
        return K.get_value(self._gamma_)
        
    @property
    def beta_(self):
        return K.get_value(self._beta_)
        
    @property
    def running_mean_(self):
        return K.get_value(self._running_mean_)

    @property
    def running_var_(self):
        return K.get_value(self._running_var_)
        
    @property
    def inner_updates_(self):
        return self._inner_updates_
        
    # layer's info & params
    @property
    def info_(self):
        kwargs = self._base_kwargs_
        info = {'class_name': self.__class__.__name__, 
                 'id': self._id_, 
                 'name': self._name_, 
                 'axis': self._axis_, 
                 'gamma': self.gamma_, 
                 'beta': self.beta_, 
                 'running_mean': self.running_mean_, 
                 'running_var': self.running_var_, 
                 'trainable_params': self._trainable_params_, 
                 'kwargs': kwargs}
        return info
        
    # ---------- Public methods ----------
    
    def compile(self):
        self._inputs_ = [layer.output_ for layer in self._prevs_]
        self._output_ = self._get_output(self._inputs_[0])
    
    @classmethod
    def load_from_info(cls, info):
        layer = cls(axis=info['axis'], gamma_init=info['gamma'], beta_init=info['beta'], 
                     running_mean_init=info['running_mean'], running_var_init=info['running_var'], 
                     trainable_params=info['trainable_params'], **info['kwargs']
                  )
        return layer
        
    # set trainable params
    def set_trainable_params(self, trainable_params):
        legal_params = ['gamma', 'beta']
        self._params_ = []
        for ch in trainable_params:
            assert ch in legal_params, "'ch' is not a param of " + self.__class__.__name__ + "! "
            self._params_.append(self.__dict__['_'+ch+'_'])
        
    # ---------- Private methods ----------
        
    # train phase
    def _tr_phase(self, input, bn_param_shape, axis):
        mean_ = K.mean(input, axis)
        var_ = K.var(input, axis)

        # get broadcast mean & var, gamma & var
        bc_mean_ = K.broadcast(x=mean_, x_ndim=len(bn_param_shape), bc_axis=axis)
        bc_var_ = K.broadcast(x=var_, x_ndim=len(bn_param_shape), bc_axis=axis)
        bc_gamma_ = K.broadcast(x=self._gamma_, x_ndim=len(bn_param_shape), bc_axis=axis)
        bc_beta_ = K.broadcast(x=self._beta_, x_ndim=len(bn_param_shape), bc_axis=axis)
        
        # get batch normalized output
        output = K.batch_normalization(input, bc_gamma_, bc_beta_, bc_mean_, bc_var_, self._eps_)
        
        # update running mean & var
        new_running_mean = K.moving_average(self._running_mean_, mean_, self._momentum_)
        new_running_var = K.moving_average(self._running_var_, var_, self._momentum_)
        
        inner_updates = [(self._running_mean_, new_running_mean), 
                          (self._running_var_, new_running_var)]

        self._inner_updates_ = inner_updates
        
        return output
        
    # test phase
    def _te_phase(self, input, bn_param_shape, axis):
        # get broadcast mean & var, gamma & var
        bc_running_mean_ = K.broadcast(x=self._running_mean_, x_ndim=len(bn_param_shape), bc_axis=axis)
        bc_running_var_ = K.broadcast(x=self._running_var_, x_ndim=len(bn_param_shape), bc_axis=axis)
        bc_gamma_ = K.broadcast(x=self._gamma_, x_ndim=len(bn_param_shape), bc_axis=axis)
        bc_beta_ = K.broadcast(x=self._beta_, x_ndim=len(bn_param_shape), bc_axis=axis)
        
        output = K.batch_normalization(input, bc_gamma_, bc_beta_, bc_running_mean_, bc_running_var_, self._eps_)
        return output
        
    # get normlize shape
    def _get_bn_param_shape(self, out_shape, axis):
        bn_param_shape = ()
        for i1 in xrange(len(out_shape)):
            if i1 not in axis:
                bn_param_shape += (out_shape[i1],)
        return bn_param_shape

    def _get_output(self, input):
        output = K.ifelse(K.eq(self._tr_phase_node_, 1.), 
                           self._tr_phase(input, self._bn_param_shape_, self._axis_), 
                           self._te_phase(input, self._bn_param_shape_, self._axis_))
        return output
    
    def _get_reg(self):
        reg_value = 0.
        # TODO "The Regularization on BN will is skipped! now"
        return reg_value
    
    
# TODO not used now. 
# '''
# reweight each channel separately
# '''
# class Reweight(Layer):
#     def __init__(self, alpha_init=None, beta_init=None, name=None):
#         super(Reweight, self).__init__(name)
#         self._alpha_init_ = alpha_init
#         self._beta_init_ = beta_init
#     
#     def __call__(self, in_layers):
#         in_layers = to_list(in_layers)
#         in_shape = in_layers[0].out_shape_
#         input = in_layers[0].output_
#         
#         self._alpha_ = self._init_params(self._alpha_init_, 'ones', shape=(in_shape[-1]), name=str(self._id_)+'_a')
#         self._beta_ = self._init_params(self._beta_init_, 'zeros', shape=(in_shape[-1]), name=str(self._id_)+'_b')
#         output = input * self._alpha_ + self._beta_
#         
#         # assign attributes
#         self._prevs_ = in_layers
#         self._nexts_ = []
#         self._out_shape_ = in_shape
#         self._output_ = output
#         self._params_ = [self._alpha_, self._beta_]
#         self._reg_value_ = 0.
#         
#         # below are compulsory parts
#         [layer.add_next(self) for layer in in_layers]     # add this layer to all prev layers' nexts pointer
#         self._check_attributes()                             # check if all attributes are implemented
#         return self
#         
#     # ---------- Public attributes ----------
#         
#     @property
#     def alpha_(self):
#         return K.get_value(self._alpha_)
#         
#     @property
#     def beta_(self):
#         return K.get_value(self._beta_)
#         
#     @property
#     def info_(self):
#         dict = {'class_name': self.__class__.__name__, 
#                  'id': self._id_, 
#                  'name': self._name_, 
#                  'alpha': self.alpha_, 
#                  'beta': self.beta_,}
#         return dict
#     
#     # ---------- Public methods ----------
#     
#     # load layer from info
#     @classmethod
#     def load_from_info(cls, info):
#         layer = cls(alpha_init=info['alpha'], beta_init=info['beta'], name=info['name'])
#         return layer