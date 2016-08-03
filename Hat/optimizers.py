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
Stochastic Grident Descend
'''
class SGD():
    def __init__( self, lr, rho ):
        self._lr = lr
        self._rho = rho
        self._vs = []
        
    def get_updates( self, params, gparams ):
        if len( self._vs ) == 0:
            for param in params:
                self._vs.append( K.shared( np.zeros_like( K.get_value( param ) ) ) )
                
        update_params = []
        update_vs = []
        
        for i1 in xrange( len(params) ):
            v_new = self._rho * self._vs[i1] + self._lr * gparams[i1]
            update_params.append( ( params[i1], params[i1] - v_new ) )
            update_vs.append( ( self._vs[i1], v_new ) )
            
        updates = update_params + update_vs
        return updates
        
        
'''
Adagrad
[1] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods for online learning and stochastic optimization." Journal of Machine Learning Research 12.Jul (2011): 2121-2159.
'''
class Adagrad():
    def __init__( self, lr=0.01, eps=1e-6 ):
        self._lr = lr
        self._eps = eps
        self._Gs = []
        
    def get_updates( self, params, gparams ):
        if len( self._Gs ) == 0:
            for param in params:
                self._Gs.append( K.shared( np.zeros_like( K.get_value( param ) ) ) )
                
        update_params = []
        update_Gs = []
        
        for i1 in xrange( len(params) ):
            G_new = self._Gs[i1] + gparams[i1]**2
            update_Gs.append( ( self._Gs[i1], G_new ) )
            update_params.append( ( params[i1], params[i1] - self._lr * gparams[i1] / K.sqrt( G_new + self._eps ) ) )
            
        return update_params + update_Gs
    
'''
Adadelta
[1] Zeiler, Matthew D. "ADADELTA: an adaptive learning rate method." arXiv preprint arXiv:1212.5701 (2012).
'''
class Adadelta():
    def __init__( self, rou=0.95, eps=1e-8 ):
        self._eps = eps
        self._rou = rou
        self._Egs = []
        self._Exs = []
        
        
    def get_updates( self, params, gparams ):
        if not self._Egs:
            for param in params:
                self._Egs += [ K.shared( np.zeros_like(param.get_value()) ) ]
                self._Exs += [ K.shared( np.zeros_like(param.get_value()) ) ]
                
        update_params = []
        update_Egs = []
        update_Exs = []
        
        for i1 in xrange( len(params) ):
            Eg_new = self._rou * self._Egs[i1] + ( 1 - self._rou ) * gparams[i1]**2
            delta_x = - np.sqrt( self._Exs[i1] + self._eps ) / np.sqrt( Eg_new + self._eps ) * gparams[i1]
            Ex_new = self._rou * self._Exs[i1] + ( 1 - self._rou ) * delta_x**2
            update_Egs += [ ( self._Egs[i1], Eg_new ) ]
            update_Exs += [ ( self._Exs[i1], Ex_new ) ]
            update_params += [ ( params[i1], params[i1] + delta_x ) ]
            
        updates = update_params + update_Egs + update_Exs
        return updates
    
'''
Rmsprop
[1] Tieleman, Tijmen, and Geoffrey Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude." COURSERA: Neural Networks for Machine Learning 4.2 (2012).
'''
class Rmsprop():
    def __init__( self, lr=0.001, rho=0.9, eps=1e-6 ):
        self._lr = lr
        self._rho = rho
        self._eps = eps
        self._Gs = []
        
    def get_updates( self, params, gparams ):
        if len( self._Gs ) == 0:
            for param in params:
                self._Gs.append( K.shared( np.zeros_like( K.get_value( param ) ) ) )
                
        update_params = []
        update_Gs = []
        
        for i1 in xrange( len(params) ):
            G_new = self._rho * self._Gs[i1] + ( 1 - self._rho ) * gparams[i1]**2
            update_Gs.append( ( self._Gs[i1], G_new ) )
            update_params.append( ( params[i1], params[i1] - self._lr * gparams[i1] / K.sqrt( G_new + self._eps ) ) )
            
        return update_params + update_Gs
        
'''
Adam
[1] Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
'''        
class Adam():
    def __init__( self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8 ):
        self._alpha = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._ms = []
        self._vs = []
        self._epoch = K.shared(1)
        
    def get_updates( self, params, gparams ):
        if not self._ms:
            for param in params:
                self._ms += [ K.shared( np.zeros_like( param.get_value() ) ) ]
                self._vs += [ K.shared( np.zeros_like( param.get_value() ) ) ]
                
        update_params = []
        update_ms = []
        update_vs = []
        
        for i1 in xrange( len(params) ):
            m_new = self._beta1 * self._ms[i1] + ( 1 - self._beta1 ) * gparams[i1]
            v_new = self._beta2 * self._vs[i1] + ( 1 - self._beta2 ) * gparams[i1]**2
            m_unbias = m_new / ( 1 - K.power( self._beta1, self._epoch ) )
            v_unbias = v_new / ( 1 - K.power( self._beta2, self._epoch ) )
            param_new = params[i1] - self._alpha * m_unbias / ( K.sqrt( v_unbias ) + self._eps )
            update_ms += [ ( self._ms[i1], m_new ) ]
            update_vs += [ ( self._vs[i1], v_new ) ]
            update_params += [ ( params[i1], param_new) ]
            
        update_epoch = [ ( self._epoch, self._epoch + 1.) ]
        
        updates = update_params + update_ms + update_vs + update_epoch
        return updates