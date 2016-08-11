'''
SUMMARY:  serialize model
AUTHOR:   Qiuqiang Kong
Created:  2016.08.01
Modified: -
--------------------------------------
'''
from supports import BFT
import cPickle
from layers.core import *
from layers.cnn import *
from layers.pool import *
from layers.rnn import *
from models import *

### save model
def save( md, path ):
    # find all nodes using BFT
    id_list, layer_list = BFT( md.in_layers_ )
    n_layers = len( id_list )
    
    # construct dicts
    atom_list = []
    for layer in layer_list:
        atom = {}
        
        # add data segment to atom
        atom['info'] = layer.info_
        
        # add next_ids to atom
        next_ids = []
        for next_layer in layer.nexts_:
            next_ids.append( next_layer.id_ )
        atom['next_ids'] = next_ids
        
        # add prev_ids to atom
        prev_ids = []
        for prev_layer in layer.prevs_:
            prev_ids.append( prev_layer.id_ )
        atom['prev_ids'] = prev_ids
        
        # add atom to list
        atom_list.append( atom )
        
    md_pure_data = {}
    md_pure_data['class_name'] = md.__class__.__name__
    md_pure_data['info'] = md.info_
    md_pure_data['atom_list'] = atom_list
        
    # dump
    cPickle.dump( md_pure_data, open( path, 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
        
### load model
def load( path ):
    # return atom from id
    def _find_atom_( atom_list, id ):
        for atom in atom_list:
            if atom['info']['id'] == id:
                return atom
    
    # return layer from id
    def _find_layer( layer_list, id ):
        for layer in layer_list:
            if layer.id_ == id:
                return layer
    
    # return all in_layers
    def _find_in_layers( layer_list ):
        in_layers = []
        for layer in layer_list:
            if not layer.prevs_:
                in_layers.append( layer )
        return in_layers
    
    # return all out_layers
    def _find_out_layers( layer_list ):
        out_layers = []
        for layer in layer_list:
            if not layer.nexts_:
                out_layers.append( layer )
        return out_layers
    
    
    # load serialized data
    md_pure_data = cPickle.load( open( path, 'rb' ) )
    
    md_info = md_pure_data['info']
    md_func = globals().get( md_pure_data['class_name'] )
    atom_list = md_pure_data['atom_list']
    
    # create layer from atom one by one
    layer_list = []
    for id in xrange( len( atom_list ) ):
        atom = _find_atom_( atom_list, id )
        LayerClass = globals().get( atom['info']['class_name'] )
        assert LayerClass is not None, "Try import '" + atom['class_name'] + "' to serializations.py!"

        if not atom['prev_ids']:
            layer = LayerClass.load_from_info( atom['info'] )
        else:
            prev_ids = atom['prev_ids']
            in_layers = [ _find_layer( layer_list, id ) for id in prev_ids ]
            layer = LayerClass.load_from_info( atom['info'] )( in_layers )

        layer_list.append( layer )
        
        
    # find in_layers & out_layers
    in_layers = [ _find_layer( layer_list, id ) for id in md_info['in_ids'] ]
    out_layers = [ _find_layer( layer_list, id ) for id in md_info['out_ids'] ]
    inter_layers = [ _find_layer( layer_list, id ) for id in md_info['inter_ids'] ]
    
    # construct model
    md = md_func.load_from_info( in_layers, out_layers, inter_layers, md_info )
    
    return md
        
        