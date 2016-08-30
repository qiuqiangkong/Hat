'''
SUMMARY:  Regularizations
AUTHOR:   Qiuqiang Kong
Created:  2016.05.20
Modified: 2016.08.02 add info(), load_from_info()
--------------------------------------
'''
from abc import ABCMeta, abstractmethod, abstractproperty
import backend as K


'''
Base class for Regularization
'''
class Regularization( object ):
    __metaclass__ = ABCMeta
     
    @abstractmethod
    def get_reg():
        pass
        
    @abstractproperty
    def info_():
        pass
    
    @abstractmethod
    def load_from_info():
        pass
    
'''
This is an instruction on how to create your own Regularization
'''
class YourReg( Regularization ):
    # See L1( Regularization ) as an example. 
    # get_reg(), info_(), load_from_info() must be implemented
    pass
    
    
###
'''
L1 regularization
'''
class L1( Regularization ):
    def __init__( self, l1=0.01 ):
        self._l1_ = l1
        
    def get_reg( self, params ):
        return self._l1_ * K.sum( [ K.sum( K.abs( param ) ) for param in params ] )
        
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'l1': self._l1_ }
        return dict
        
    @classmethod
    def load_from_info( cls, info ):
        reg = cls( l1=info['l1'] )
        return reg

'''
L2 regularization
'''        
class L2( Regularization ):
    def __init__( self, l2=0.01 ):
        self._l2_ = l2
        
    def get_reg( self, params ):
        return self._l2_ * K.sum( [ K.sum( K.sqr( param ) ) for param in params ] )
        
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'l2': self._l2_ }
        return dict
        
    @classmethod
    def load_from_info( cls, info ):
        reg = cls( l2=info['l2'] )
        return reg
        
'''
L1 & L2 regularization
'''
class L1L2( Regularization ):
    def __init__( self, l1=0.01, l2=0.01 ):
        self._l1_ = l1
        self._l2_ = l2
        
    def get_reg( self, params ):
        reg = self._l1_ * K.sum( [ K.sum( K.abs( param ) ) for param in params ] ) + \
              self._l2_ * K.sum( [ K.sum( K.sqr( param ) ) for param in params ] )
        return reg
        
    @property
    def info_( self ):
        dict = { 'class_name': self.__class__.__name__, 
                 'l1': self._l1_, 
                 'l2': self._l2_ }
        return dict
        
    @classmethod
    def load_from_info( cls, info ):
        reg = cls( l1=info['l1'], l2=info['l2'] )
        return reg
        
### return info from reg_obj
def get_info( reg_obj ):
    if reg_obj is None:
        return None
    else:
        return reg_obj.info_
        
### return reg_obj from info
def get_obj( reg_info ):
    if reg_info is None:
        return None
    else:
        return get( reg_info['class_name'] ).load_from_info( reg_info )
        
### return class from name
def get( reg_type ):
    f = globals().get( reg_type )
    if f is None:
        raise Exception( "Regularization " + reg_type + " does not exist! You should define it before using!" )
    else:
        return f
        
### register user defined regularization
def register( reg ):
    exec( reg.__name__ + " = reg", locals(), globals() )