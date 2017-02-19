"""
SUMMARY:  optimization methods. Modified from my previous optimization.
AUTHOR:   Qiuqiang Kong
Created:  2016.05.20
Modified: 2016.08.02 modify name sh_variable to shared
--------------------------------------
"""
import numpy as np
import backend as K



class Base(object):
    """
    Base class for optimization classes. 
    """
    def _reset_memory(self, memory):
        """Reset the memory to zero. 
        """
        for i1 in xrange(len(memory)):
            K.set_value(memory[i1], np.zeros_like(memory[i1].get_value()))
            

class SGD(Base):
    """ Stochastic Grident Descend
    """
    def __init__(self, lr, momentum):
        self._lr_ = lr
        self._momentum_ = momentum
    
    def get_updates(self, params, gparams):
        self._vs_ = []
        for param in params:
            self._vs_.append(K.shared(np.zeros_like(K.get_value(param))))
                
        update_params = []
        update_vs = []
        
        for p, g, a in zip(params, gparams, self._vs_):
            a_new = self._momentum_ * a + self._lr_ * g
            p_new = p - a_new
            update_params.append((p, p_new))
            update_vs.append((a, a_new))
            
        updates = update_params + update_vs
        return updates
        
    def reset(self):
        self._reset_memory(self._vs_)
        

class Adagrad(Base):
    """ Adagrad
    Ref: Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods 
         for online learning and stochastic optimization." Journal of Machine 
         Learning Research 12.Jul (2011): 2121-2159.
    """
    def __init__(self, lr=0.01, eps=1e-8):
        self._lr_ = lr
        self._eps_ = eps
    
    def get_updates(self, params, gparams):
        self._accumulators_ = []
        for param in params:
            self._accumulators_.append(K.shared(np.zeros_like(K.get_value(param))))
                
        updates = []
        
        for p, g, a in zip(params, gparams, self._accumulators_):
            a_new = a + K.sqr(g)
            p_new = p - self._lr_ * g / (K.sqrt(a_new) + self._eps_)
            updates.append((a, a_new))
            updates.append((p, p_new))
        
        return updates
    
    def reset(self):
        self._reset_memory(self._accumulators_)


class Adadelta(Base):
    """ Adadelta
    Ref: Zeiler, Matthew D. "ADADELTA: an adaptive learning rate method." 
         arXiv preprint arXiv:1212.5701 (2012).
    """
    def __init__(self, rou=0.95, eps=1e-8):
        self._eps_ = eps
        self._rou_ = rou

    def get_updates(self, params, gparams):
        self._accumulators_ = []
        self._delta_accumulators_ = []
        for param in params:
            self._accumulators_ += [K.shared(np.zeros_like(param.get_value()))]
            self._delta_accumulators_ += [K.shared(np.zeros_like(param.get_value()))]
                
        updates = []

        for p, g, a, d_a in zip(params, gparams, self._accumulators_, self._delta_accumulators_):
            a_new = self._rou_ * a + (1. - self._rou_) * K.sqr(g)
            updates.append((a, a_new))
            
            p_delta = - g * K.sqrt(d_a + self._eps_) / K.sqrt(a_new + self._eps_)
            p_new = p + p_delta
            updates.append((p, p_new))
            
            d_a_new = self._rou_ * d_a + (1. - self._rou_) * K.sqr(p_delta)
            updates.append((d_a, d_a_new))
        
        return updates
        
    def reset(self):
        self._reset_memory(self._accumulators_)
        self._reset_memory(self._delta_accumulators_)
    

class Rmsprop(Base):
    """Rmsprop
    Ref: Tieleman, Tijmen, and Geoffrey Hinton. "Lecture 6.5-rmsprop: Divide the 
         gradient by a running average of its recent magnitude." COURSERA: 
         Neural Networks for Machine Learning 4.2 (2012).
    """
    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        super(Rmsprop, self).__init__()
        self._lr_ = lr
        self._rho_ = rho
        self._eps_ = eps
         
    def get_updates(self, params, gparams):
        self._accumulators_ = []
        for param in params:
            self._accumulators_.append(K.shared(np.zeros_like(K.get_value(param))))
                
        updates = []
        
        for p, g, a in zip(params, gparams, self._accumulators_):
            a_new = self._rho_ * a + (1 - self._rho_) * K.sqr(g)
            p_new = p - self._lr_ * g / (K.sqrt(a_new) + self._eps_)
            updates.append((p, p_new))
            updates.append((a, a_new))
            
        return updates
        
    def reset(self):
        self._reset_memory(self._accumulators_)
        
        
# # Slow version
# class Adam(Base):
#     def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
#         self._alpha_ = lr
#         self._beta1_ = beta1
#         self._beta2_ = beta2
#         self._eps_ = eps
#         self._epoch_ = K.shared(1)
#         
#         
#     def get_updates(self, params, gparams):
#         self._ms_ = []
#         self._vs_ = []
#         for param in params:    
#             self._ms_ += [K.shared(np.zeros_like(param.get_value()))]
#             self._vs_ += [K.shared(np.zeros_like(param.get_value()))]
#                     
#         update_params = []
#         update_ms = []
#         update_vs = []
#         
#         for i1 in xrange(len(params)):
#             m_new = self._beta1_ * self._ms_[i1] + (1 - self._beta1_) * gparams[i1]
#             v_new = self._beta2_ * self._vs_[i1] + (1 - self._beta2_) * gparams[i1]**2
#             m_unbias = m_new / (1 - K.power(self._beta1_, self._epoch_))
#             v_unbias = v_new / (1 - K.power(self._beta2_, self._epoch_))
#             param_new = params[i1] - self._alpha_ * m_unbias / (K.sqrt(v_unbias) + self._eps_)
#             update_ms += [(self._ms_[i1], m_new)]
#             update_vs += [(self._vs_[i1], v_new)]
#             update_params += [(params[i1], param_new)]
#             
#         update_epoch = [(self._epoch_, self._epoch_ + 1.)]
#         
#         updates = update_params + update_ms + update_vs + update_epoch
#         return updates
#         
#         
#     # reset memories to zero, epoch to 1
#     def reset(self):
#         self._reset_memory(self._ms_)
#         self._reset_memory(self._vs_)
#         K.set_value(self._epoch_, 1)


# Fast version
class Adam(Base):
    """ Adam
    Ref: Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization." 
         arXiv preprint arXiv:1412.6980 (2014).
    """       
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self._alpha_ = lr
        self._beta1_ = beta1
        self._beta2_ = beta2
        self._eps_ = eps
        self._iter_ = K.shared(0)
        
    def get_updates(self, params, gparams):
        self._ms_ = []
        self._vs_ = []
        for param in params:    
            self._ms_ += [K.shared(np.zeros_like(param.get_value()))]
            self._vs_ += [K.shared(np.zeros_like(param.get_value()))]

        updates = []
        
        t = self._iter_ + 1
        alpha_t = self._alpha_ * (K.sqrt(1. - K.power(self._beta2_, t)) / (1. - K.power(self._beta1_, t)))
        
        for p, g, m, v in zip(params, gparams, self._ms_, self._vs_):
            m_new = self._beta1_ * m + (1. - self._beta1_) * g
            updates.append((m, m_new))
            
            v_new = self._beta2_ * v + (1. - self._beta2_) * K.sqr(g)
            updates.append((v, v_new))
            
            p_new = p - alpha_t * m_new / (K.sqrt(v_new) + self._eps_)
            updates.append((p, p_new))
            
        updates.append((self._iter_, self._iter_ + 1))
        
        return updates
        
    def reset(self):
        self._reset_memory(self._ms_)
        self._reset_memory(self._vs_)
        K.set_value(self._iter_, 0)