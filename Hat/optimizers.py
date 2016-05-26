'''
SUMMARY:  optimization methods. Modified from my previous optimization.
AUTHOR:   Qiuqiang Kong
Created:  2016.05.20
Modified: -
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
                self._vs.append( K.sh_variable( np.zeros_like( K.get_value( param ) ) ) )
                
        update_params = []
        update_vs = []
        
        vs_new = []
        for i1 in xrange( len(params) ):
            vs_new.append( self._rho * self._vs[i1] + self._lr * gparams[i1] )
            update_params.append( ( params[i1], params[i1] - vs_new[i1] ) )
            update_vs.append( ( self._vs[i1], vs_new[i1] ) )
            
        updates = update_params + update_vs
        return updates
        
        
'''
Adagrad
'''
class Adagrad():
    def __init__( self, lr=0.01 ):
        self._lr = lr
        self._Gs = []
        self._eps = 1e-6
        
    def get_updates( self, params, gparams ):
        if len( self._Gs ) == 0:
            for param in params:
                self._Gs.append( K.sh_variable( np.zeros_like( K.get_value( param ) ) ) )
                
        update_params = []
        update_Gs = []
        
        Gs_new = []
        for i1 in xrange( len(params) ):
            Gs_new.append( self._Gs[i1] + gparams[i1]**2 )
            update_Gs.append( ( self._Gs[i1], Gs_new[i1] ) )
            
        for i1 in xrange( len(params) ):
            update_params.append( ( params[i1], params[i1] - self._lr * gparams[i1] / K.sqrt( Gs_new[i1] + self._eps ) ) )
            
        return update_params + update_Gs
    
'''
Rmsprop, using default config is a good choice. 
'''
class Rmsprop():
    def __init__( self, lr=0.001, rho=0.9 ):
        self._lr = lr
        self._rho = rho
        self._Gs = []
        self._eps = 1e-6
        
    def get_updates( self, params, gparams ):
        if len( self._Gs ) == 0:
            for param in params:
                self._Gs.append( K.sh_variable( np.zeros_like( K.get_value( param ) ) ) )
                
        update_params = []
        update_Gs = []
        
        Gs_new = []
        for i1 in xrange( len(params) ):
            Gs_new.append( self._rho * self._Gs[i1] + ( 1 - self._rho ) * gparams[i1]**2 )
            update_Gs.append( ( self._Gs[i1], Gs_new[i1] ) )
            
        for i1 in xrange( len(params) ):
            update_params.append( ( params[i1], params[i1] - self._lr * gparams[i1] / K.sqrt( Gs_new[i1] + self._eps ) ) )
            
        return update_params + update_Gs