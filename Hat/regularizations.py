import backend as K

class L1():
    def __init__( self, l1=0.01 ):
        self._l1 = l1
        
    def reg_value( self, params ):
        return self._l1 * K.sum( [ K.sum( K.abs( param ) ) for param in params ] )
        
class L2():
    def __init__( self, l2=0.01 ):
        self._l2 = l2
        
    def reg_value( self, params ):
        return self._l2 * K.sum( [ K.sum( K.sqr( param ) ) for param in params ] )
        
class L1L2():
    def __init__( self, l1=0.01, l2=0.01 ):
        self._l1 = l1
        self._l2 = l2
        
    def reg_value( self, params ):
        reg = self._l1 * K.sum( [ K.sum( K.abs( param ) ) for param in params ] ) + \
              self._l2 * K.sum( [ K.sum( K.sqr( param ) ) for param in params ] )
        return reg