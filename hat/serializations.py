'''
SUMMARY:  serialize model
AUTHOR:   Qiuqiang Kong
Created:  2016.08.01
Modified: -
--------------------------------------
'''
# from supports import BFT
import cPickle
from layers.core import *
from layers.cnn import *
from layers.pooling import *
from layers.rnn import *
from layers.normalization import *
from layers.embeddings import *
from models import *


def get_model_data_seg(md):
    """Get data segment of model
    """
    effective_layers= md.get_effective_layers(md.in_layers_, md.out_layers_)
    
    atom_list = []
    
    for layer in effective_layers:
        atom = {}
        atom['info'] = layer.info_
        
        # pointer to next layers
        next_ids = []
        for next_layer in layer.nexts_:
            next_ids.append(next_layer.id_)
        atom['next_ids'] = next_ids
        
        # pointer to prev layers
        prev_ids = []
        for prev_layer in layer.prevs_:
            prev_ids.append(prev_layer.id_)
        atom['prev_ids'] = prev_ids
        
        atom_list.append(atom)
    
    md_data_seg = {}
    md_data_seg['class_name'] = md.__class__.__name__
    md_data_seg['info'] = md.info_
    md_data_seg['atom_list'] = atom_list
    
    return md_data_seg
    

def save(md, path):
    """Save model
    """
    # get data segment of model
    md_data_seg = get_model_data_seg(md)
        
    # dump
    cPickle.dump(md_data_seg, open(path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        

def load(path):
    """load model
    """
    # search layer by id
    def _find_layer(layer_list, id):
        for layer in layer_list:
            if layer.id_ == id:
                return layer
    
    # load serialized data
    md_data_seg = cPickle.load(open(path, 'rb'))
    
    md_info = md_data_seg['info']
    md_func = globals().get(md_data_seg['class_name'])
    atom_list = md_data_seg['atom_list']
    
    layer_list = []
    for atom in atom_list:
        LayerClass = globals().get(atom['info']['class_name'])
        assert LayerClass is not None, "Try serializations.register(" + atom['info']['class_name'] + ") before loading model!"
        
        if not atom['prev_ids']:
            layer = LayerClass.load_from_info(atom['info'])
        else:
            prev_ids = atom['prev_ids']
            in_layers = [_find_layer(layer_list, id) for id in prev_ids]
            layer = LayerClass.load_from_info(atom['info'])(in_layers)
    
        layer_list.append(layer)
 
    # find in_layers & out_layers
    in_layers = [_find_layer(layer_list, id) for id in md_info['in_ids']]
    out_layers = [_find_layer(layer_list, id) for id in md_info['out_ids']]
    inter_layers = [_find_layer(layer_list, id) for id in md_info['any_ids']]
    
    # construct model
    md = md_func.load_from_info(in_layers, out_layers, inter_layers, md_info)
    md.compile()
    
    return md
        

def register(layer):
    """Register user defined layer
    
    Args:
      layer: user defined layer. 
    """
    exec(layer.__name__ + " = layer", locals(), globals())
        