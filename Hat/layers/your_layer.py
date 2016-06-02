from core import Layer
from ..import backend as K

'''
# DO NOT DELETE! 
# You can build your own computation layer based on Lambda layer. 
# computation layers should not have parameters. 
'''
# Eg. 
def your_func1( input, in_shape, **kwargs ):
    # your implementation here
    # assert 'xxx' in kwargs, "You must specifiy xxx kwarg!"
    # ...
    return output, out_shape
    
def your_func2( input1, input2, in_shape1, in_shape2, **kwargs ):
    # your implementation here
    # assert 'xxx' in kwargs, "You must specifiy xxx kwarg!"
    # ...
    return output, out_shape

# Usage
# md.add( Lambda( your_func1 ) )

### Or you can sealed it as a class
# Eg. 
# without kwargs
class YourLayer1( Lambda ):
    def __init__( self, name=None ):
        super( YourLayer, self ).__init__( your_func1, name )
        
# with kwargs
class YourLayer2( Lambda ):
    def __init__( self, name=None, **kwargs ):
        super( YourLayer, self ).__init__( your_func1, name, **kwargs )