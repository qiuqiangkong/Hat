'''
SUMMARY:  optimization methods. Modified from my previous optimization.
AUTHOR:   Qiuqiang Kong
Created:  2016.05.20
Modified: 2016.08.02 modify name sh_variable to shared
--------------------------------------
'''
import numpy as np
import backend as K

'''
Base class for optimization classes
'''
class Base( object ):
    # reset the memory
    def _reset_memory( self, memory ):
        for i1 in xrange( len( memory ) ):
            K.set_value( memory[i1], np.zeros_like( memory[i1].get_value() ) )
            

'''
Stochastic Grident Descend
'''
class SGD( Base ):
    def __init__( self, lr, rho ):
        self._lr_ = lr
        self._rho_ = rho
        
    def get_updates( self, params, gparams ):
        self._vs_ = []
        for param in params:
            self._vs_.append( K.shared( np.zeros_like( K.get_value( param ) ) ) )
                
        update_params = []
        update_vs = []
        
        for i1 in xrange( len(params) ):
            v_new = self._rho_ * self._vs_[i1] + self._lr_ * gparams[i1]
            update_params.append( ( params[i1], params[i1] - v_new ) )
            update_vs.append( ( self._vs_[i1], v_new ) )
            
        updates = update_params + update_vs
        return updates
        
    # reset memories to zero
    def reset( self ):
        self._reset_memory( self._vs_ )
        

'''
Adagrad
[1] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods for online learning and stochastic optimization." Journal of Machine Learning Research 12.Jul (2011): 2121-2159.
'''
class Adagrad( Base ):
    def __init__( self, lr=0.01, eps=1e-6 ):
        self._lr_ = lr
        self._eps_ = eps
        
    def get_updates( self, params, gparams ):
        self._Gs_ = []
        for param in params:
            self._Gs_.append( K.shared( np.zeros_like( K.get_value( param ) ) ) )
                
        update_params = []
        update_Gs = []
        
        for i1 in xrange( len(params) ):
            G_new = self._Gs_[i1] + gparams[i1]**2
            update_Gs.append( ( self._Gs_[i1], G_new ) )
            update_params.append( ( params[i1], params[i1] - self._lr_ * gparams[i1] / K.sqrt( G_new + self._eps_ ) ) )
            
        return update_params + update_Gs
    
    # reset memories to zero
    def reset( self ):
        self._reset_memory( self._Gs_ )
    
    
'''
Adadelta
[1] Zeiler, Matthew D. "ADADELTA: an adaptive learning rate method." arXiv preprint arXiv:1212.5701 (2012).
'''
class Adadelta( Base ):
    def __init__( self, rou=0.95, eps=1e-8 ):
        self._eps_ = eps
        self._rou_ = rou

    def get_updates( self, params, gparams ):
        self._Egs_ = []
        self._Exs_ = []
        for param in params:
            self._Egs_ += [ K.shared( np.zeros_like(param.get_value()) ) ]
            self._Exs_ += [ K.shared( np.zeros_like(param.get_value()) ) ]
                
        update_params = []
        update_Egs = []
        update_Exs = []
        
        for i1 in xrange( len(params) ):
            Eg_new = self._rou_ * self._Egs_[i1] + ( 1 - self._rou_ ) * gparams[i1]**2
            delta_x = - np.sqrt( self._Exs_[i1] + self._eps_ ) / np.sqrt( Eg_new + self._eps_ ) * gparams[i1]
            Ex_new = self._rou_ * self._Exs_[i1] + ( 1 - self._rou_ ) * delta_x**2
            update_Egs += [ ( self._Egs_[i1], Eg_new ) ]
            update_Exs += [ ( self._Exs_[i1], Ex_new ) ]
            update_params += [ ( params[i1], params[i1] + delta_x ) ]
            
        updates = update_params + update_Egs + update_Exs
        return updates
        
    # reset memories to zero
    def reset( self ):
        self._reset_memory( self._Egs_ )
        self._reset_memory( self._Exs_ )
    
    
'''
Rmsprop
[1] Tieleman, Tijmen, and Geoffrey Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude." COURSERA: Neural Networks for Machine Learning 4.2 (2012).
'''
class Rmsprop( Base ):
    def __init__( self, lr=0.001, rho=0.9, eps=1e-6 ):
        super( Rmsprop, self ).__init__()
        self._lr_ = lr
        self._rho_ = rho
        self._eps_ = eps
        
        
    def get_updates( self, params, gparams ):
        self._Gs_ = []
        for param in params:
            self._Gs_.append( K.shared( np.zeros_like( K.get_value( param ) ) ) )
                
        update_params = []
        update_Gs = []
        
        for i1 in xrange( len(params) ):
            G_new = self._rho_ * self._Gs_[i1] + ( 1 - self._rho_ ) * gparams[i1]**2
            update_Gs.append( ( self._Gs_[i1], G_new ) )
            update_params.append( ( params[i1], params[i1] - self._lr_ * gparams[i1] / K.sqrt( G_new + self._eps_ ) ) )
            
        return update_params + update_Gs
        
        
    # reset memories to zero
    def reset( self ):
        self._reset_memory( self._Gs_ )
        
        
'''
Adam
[1] Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
'''        
class Adam( Base ):
    def __init__( self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8 ):
        self._alpha_ = lr
        self._beta1_ = beta1
        self._beta2_ = beta2
        self._eps_ = eps
        self._epoch_ = K.shared(1)
        
        
    def get_updates( self, params, gparams ):
        self._ms_ = []
        self._vs_ = []
        for param in params:    
            self._ms_ += [ K.shared( np.zeros_like( param.get_value() ) ) ]
            self._vs_ += [ K.shared( np.zeros_like( param.get_value() ) ) ]
                    
        update_params = []
        update_ms = []
        update_vs = []
        
        for i1 in xrange( len(params) ):
            m_new = self._beta1_ * self._ms_[i1] + ( 1 - self._beta1_ ) * gparams[i1]
            v_new = self._beta2_ * self._vs_[i1] + ( 1 - self._beta2_ ) * gparams[i1]**2
            m_unbias = m_new / ( 1 - K.power( self._beta1_, self._epoch_ ) )
            v_unbias = v_new / ( 1 - K.power( self._beta2_, self._epoch_ ) )
            param_new = params[i1] - self._alpha_ * m_unbias / ( K.sqrt( v_unbias ) + self._eps_ )
            update_ms += [ ( self._ms_[i1], m_new ) ]
            update_vs += [ ( self._vs_[i1], v_new ) ]
            update_params += [ ( params[i1], param_new) ]
            
        update_epoch = [ ( self._epoch_, self._epoch_ + 1.) ]
        
        updates = update_params + update_ms + update_vs + update_epoch
        return updates
        
        
    # reset memories to zero, epoch to 1
    def reset( self ):
        self._reset_memory( self._ms_ )
        self._reset_memory( self._vs_ )
        K.set_value( self._epoch_, 1 )