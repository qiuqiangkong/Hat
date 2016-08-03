'''
SUMMARY:  Build Model
AUTHOR:   Qiuqiang Kong
Created:  2016.05.01
Modified: 2016.07.25 Modify fit() to batch version
          2016.07.29 Replace [[]] * n with [ [] for e in self._out_nodes ]
--------------------------------------
'''
import sys
from supports import to_list, BFT, shuffle, memory_usage
from optimizers import *
from globals import reset_id_to_zero
import objectives as obj
import backend as K
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

class Base( object ):
    def __init__( self, in_layers ):
        # reset global id
        reset_id_to_zero()
        
        # inlayers
        in_layers = to_list( in_layers )
        self._in_layers_ = in_layers
        self._in_nodes_ = [ layer.output_ for layer in self._in_layers_ ]
        
        # find all nodes using BFT
        self._id_list_, self._layer_list_ = BFT( self._in_layers_ )
        
        # get params by traveling all layers
        self._params_ = []
        for layer in self._layer_list_:
            self._params_ += layer.params_
            
        # sum regs by traveling all layers
        self._reg_value_ = 0.
        for layer in self._layer_list_:
            self._reg_value_ += layer.reg_value_
            
        # tr_phase_node
        self._tr_phase_node_ = K.common_tr_phase_node
            
        # time
        self._tr_time_ = 0.
        self._epoch_ = 0
    
    # print progress on screen
    def _print_progress( self, epoch, batch_num, curr_batch_num ):
        sys.stdout.write("%d-th epoch %d%%   \r" % ( epoch, float(curr_batch_num)/float(batch_num)*100 ) )
        sys.stdout.flush()
        
    @property
    def in_layers_( self ):
        return self._in_layers_
        
    
    @property
    def in_nodes_( self ):
        return self._in_nodes_
        
    @property
    def out_nodes_( self ):
        return self._out_nodes_
        
    @property
    def gt_nodes_( self ):
        return self._gt_nodes_
    
    
    @property
    def epoch_( self ):
        return self._epoch_
        
    @property
    def tr_time_( self ):
        return self._tr_time_
        
    @property
    def tr_phase_node_( self ):
        return self._tr_phase_node_
        
    def summary( self ):
        print '---------- summary -----------'
        f = "{0:<20} {1:<20} {2:<20}"
        print f.format( 'layer_name', 'out_shape', 'n_params' )
        for layer in self._layer_list_:
            n_params = self._num_of_params( layer )
            print f.format( layer.name_, str(layer.out_shape_), str(n_params) )
        print 
            
    # get number of params in a layer
    def _num_of_params( self, layer ):
        n_params = 0
        for param in layer.params_:
            n_params += np.prod( K.get_value( param ).shape )
        return n_params
        
    # check if the dim of output is correct
    def _check_data( self, y, loss_type ):
        if loss_type in ['categorical_crossentropy', 'binary_crossentropy', 'kl_divergence']:
            for e in y:
                assert e.ndim!=1, "your y.ndim is 1, try use sparse_to_categorical(y)"
        

    def plot_connection( self ):
        G = nx.DiGraph()
        
        # add node
        for id in self._id_list_:
            G.add_node( id )
        
        # add connection
        for layer in self._layer_list_:
            for next in layer.nexts_:
                G.add_edge( layer.id_, next.id_ )
                
        # pos & labels
        pos = nx.spring_layout(G)
        labels = {}
        for layer in self._layer_list_:
            if hasattr( layer, '_act' ):
                labels[layer.id_] = layer._name_
            else:
                labels[layer.id_] = layer._name_

        # plot
        nx.draw_networkx_nodes( G , pos, node_size=800, node_color='r', node_shape='o' )
        nx.draw_networkx_labels( G, pos, labels=labels )
        nx.draw_networkx_edges( G, pos )
        plt.show()

'''
Supervised Model
'''
class Model( Base ):
    def __init__( self, in_layers, out_layers, obj_weights=None ):
        super( Model, self ).__init__( in_layers )

        # default obj_weights
        if obj_weights is None: obj_weights = [1. / len(out_layers)] * len(out_layers)
        assert len(out_layers)==len(obj_weights), "num of out_layers must equal num of obj_weights!"
        
        # out layers
        out_layers = to_list( out_layers )
        self._out_layers_ = out_layers
        self._obj_weights_ = obj_weights
        
        # out_nodes & create gt_nodes
        self._out_nodes_ = [ layer.output_ for layer in self._out_layers_ ]
        self._gt_nodes_ = [ K.placeholder( len(layer.out_shape_) ) for layer in self._out_layers_ ]

    '''
    memory mode 0 (default): transfer from cpu to gpu every time, 1: store all data in gpu
    '''
    def fit( self, x, y, batch_size=100, n_epoch=10, loss_type='categorical_crossentropy', optimizer=SGD( lr=0.01, rho=0.9 ), clip=None, callbacks=[], memory_mode=0, verbose=1 ):
        x = to_list( x )
        y = to_list( y )
        
        # format
        x = [ K.format_data(e) for e in x ]
        y = [ K.format_data(e) for e in y ]
        
        # shuffle data
        x, y = shuffle( x, y )
        
        # check data
        self._check_data( y, loss_type )
        
        # memory usage
        #mem_usage = memory_usage( x, y )
        #print 'memory usage:', mem_usage / 8e6, 'Mb'
        
        # loss
        loss_node = sum( [ obj.get( loss_type )( pred_node, gt_node ) * w 
                        for pred_node, gt_node, w in zip( self._out_nodes_, self._gt_nodes_, self._obj_weights_ ) ] )
        
        # gradient
        gparams = K.grad( loss_node + self._reg_value_, self._params_ )
        
        # todo clip gradient
        if clip is not None:
            gparams = [ K.clip( gparam, -clip, clip ) for gparam in gparams ]
        
        # gradient based opt
        updates = optimizer.get_updates( self._params_, gparams )
        
        # compile for callback
        if callbacks is not None:
            callbacks = to_list( callbacks )
            for callback in callbacks:
                callback.compile( self ) 
        
        if memory_mode==0:
            # compile model
            input_nodes = self._in_nodes_ + self._gt_nodes_
            output_nodes = [ loss_node ]
            f = K.function_no_given( input_nodes, self._tr_phase_node_, output_nodes, updates )
            
        if memory_mode==1:
            # store data in shared memory (GPU)
            sh_x = [ K.shared( value=e, name='tr_x' ) for e in x ]
            sh_y = [ K.shared( value=e, name='tr_y' ) for e in y ]
            
            # compile model
            input_nodes = self._in_nodes_ + self._gt_nodes_
            output_nodes = [ loss_node ]
            given_nodes = sh_x + sh_y
            f = K.function_given( batch_size, input_nodes, self._tr_phase_node_, output_nodes, given_nodes, updates )

        # train
        N = len( x[0] )
        batch_num = int( np.ceil( float(N) / batch_size ) )
        while self._epoch_ < n_epoch:
            # callback
            for callback in callbacks:
                if ( self._epoch_ % callback.call_freq == 0 ):
                    callback.call()
                    
            # train
            t1 = time.time()
            for i2 in xrange(batch_num):
                if memory_mode==0:
                    batch_x = [ e[i2*batch_size : min( (i2+1)*batch_size, N ) ] for e in x ]
                    batch_y = [ e[i2*batch_size : min( (i2+1)*batch_size, N ) ] for e in y ]
                    in_list = batch_x + batch_y + [1.]
                    loss = f( *in_list )[0]                     # training phase          
                if memory_mode==1:
                    loss = f(i2, 1.)[0]                     # training phase          
                if verbose: self._print_progress( self._epoch_, batch_num, i2 )
            t2 = time.time()
            self._tr_time_ += (t2 - t1)
            self._epoch_ += 1
            print '\n', 'train time: ', "%.2f " % (t2-t1), 's'          # print an empty line

    def predict( self, x, batch_size=100 ):
        # format data
        x = to_list( x )
        x = [ K.format_data(e) for e in x ]
        
        # compile predict model
        if not hasattr( self, '_f_predict' ):
            #if len( self._out_nodes_ )==1:   # if only 1 out_node, then return it directly instead of list
                #self._f_predict = K.function_no_given( self._in_nodes_, self._tr_phase_node_, self._out_nodes_[0] )
            #else:
            self._f_predict = K.function_no_given( self._in_nodes_, self._tr_phase_node_, self._out_nodes_ )
        
        # do predict
        # put all data in GPU
        if batch_size is None:
            in_list = x + [0.]
            y_out = self._f_predict( *in_list )
        # put batch data in GPU
        else:
            N = len(x[0])
            batch_num = int( np.ceil( float(N) / batch_size ) )
            n_out_nodes = len( self._out_nodes_ )
            y_out = [ [] for e in self._out_nodes_ ]
            for i1 in xrange( batch_num ):
                in_list = [ e[i1*batch_size : min( (i1+1)*batch_size, N ) ] for e in x ] + [0.]
                batch_y_out = self._f_predict( *in_list )
                for j1 in xrange(n_out_nodes):
                    y_out[j1].append( batch_y_out[j1] )
                    
            # get y_out
            y_out = [ np.concatenate(e, axis=0) for e in y_out ]
        
        if len(y_out)==1:
            return y_out[0]
        else:
            return y_out
        
    @property
    def info_( self ):
        dict = { 'obj_weights': self._obj_weights_ }
        return dict
        
    @classmethod
    def load_from_info( cls, in_layers, out_layers, info ):
        md = cls( in_layers, out_layers, info['obj_weights'] )
        return md
        
        
class Sequential( Model ):
    def __init__( self ):
        self._layer_seq_ = []
        
    def add( self, layer ):
        self._layer_seq_.append( layer )
        if len( self._layer_seq_ ) > 1:
            self._layer_seq_[-1].__call__( self._layer_seq_[-2] )

    def combine( self ):
        # regenerate model when add a new layer
        md = Model( [ self._layer_seq_[0] ], [ self._layer_seq_[-1] ] )
        return md
    
class Pretrain():
    pass