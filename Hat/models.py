import sys
from supports import to_list, BFT, shuffle, memory_usage
from optimizers import *
import objectives as obj
import backend as K
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

class Base( object ):
    def __init__( self, in_layers ):
        in_layers = to_list( in_layers )
        # inlayers
        self._in_layers = in_layers
        self._in_nodes = [ layer.output for layer in self._in_layers ]
        
        # find all nodes using BFT
        self._id_list, self._layer_list = BFT( self._in_layers )
        
        # get params by traveling all layers
        self._params = []
        for layer in self._layer_list:
            self._params += layer.params
            
        # sum regs by traveling all layers
        self._reg_value = 0.
        for layer in self._layer_list:
            self._reg_value += layer.reg_value
            
        # tr_phase_node
        self._tr_phase_node = K.common_tr_phase_node
            
        # time
        self._tr_time = 0
        self._epoch = 0
    
    # print progress on screen
    def print_progress( self, epoch, batch_num, curr_batch_num ):
        sys.stdout.write("%d-th epoch %d%%   \r" % ( epoch, float(curr_batch_num)/float(batch_num)*100 ) )
        sys.stdout.flush()
    
    # dump model    
    def dump( self, path ):
        pickle.dump( self, open( path, 'wb' ) )
        
    @property
    def in_nodes( self ):
        return self._in_nodes
        
    @property
    def out_nodes( self ):
        return self._out_nodes
        
    @property
    def gt_nodes( self ):
        return self._gt_nodes
        
    @property
    def epoch( self ):
        return self._epoch
        
    @property
    def tr_time( self ):
        return self._tr_time
        
    @property
    def tr_phase_node( self ):
        return self._tr_phase_node
        
    def summary( self ):
        print '---------- summary -----------'
        f = "{0:<20} {1:<20} {2:<20}"
        print f.format( 'layer_name', 'out_shape', 'n_params' )
        for layer in self._layer_list:
            n_params = self._num_of_params( layer )
            print f.format( layer.name, str(layer.out_shape), str(n_params) )
        print 
            
    # get number of params in a layer
    def _num_of_params( self, layer ):
        n_params = 0
        for param in layer.params:
            n_params += np.prod( K.get_value( param ).shape )
        return n_params
        
    # check if the dim of output is correct
    def _check_data( self, y, loss_type ):
        if loss_type in ['categorical_crossentropy', 'binary_crossentropy', 'kl_divergence']:
            for e in y:
                assert e.ndim==2, "your y.ndim!=2, try use sparse_to_categorical(y)"
        

    def plot_connection( self ):
        G = nx.DiGraph()
        
        # add node
        for id in self._id_list:
            G.add_node( id )
        
        # add connection
        for layer in self._layer_list:
            for next in layer.nexts:
                G.add_edge( layer.id, next.id )
                
        # pos & labels
        pos = nx.spring_layout(G)
        labels = {}
        for layer in self._layer_list:
            if hasattr( layer, '_act' ):
                labels[layer.id] = layer._name
            else:
                labels[layer.id] = layer._name

        # plot
        nx.draw_networkx_nodes( G , pos, node_size=800, node_color='r', node_shape='o' )
        nx.draw_networkx_labels( G, pos, labels=labels )
        nx.draw_networkx_edges( G, pos )
        plt.show()

'''
Supervised Model
'''
class Model( Base ):
    def __init__( self, in_layers, out_layers, obj_weights=[1.] ):
        super( Model, self ).__init__( in_layers )
        assert len(out_layers)==len(obj_weights), "num of out_layers must equal num of obj_weights!"
        
        # out layers
        out_layers = to_list( out_layers )
        self._out_layers = out_layers
        self._obj_weights = obj_weights
        
        # out_nodes & create gt_nodes
        self._out_nodes = [ layer.output for layer in self._out_layers ]
        self._gt_nodes = [ K.placeholder( len(layer.out_shape) ) for layer in self._out_layers ]
        
        
        
    '''
    Fit model. x, y can be list of ndarrays. 
    '''
    def fit( self, x, y, batch_size=100, n_epoch=10, loss_type='categorical_crossentropy', optimizer=SGD( lr=0.01, rho=0.9 ), clip=None, 
             callbacks=[], verbose=1 ):
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
        mem_usage = memory_usage( x, y )
        print 'memory usage:', mem_usage / 8e6, 'Mb'
        
        # store data in shared memory (GPU)
        sh_x = [ K.sh_variable( value=e, name='tr_x' ) for e in x ]
        sh_y = [ K.sh_variable( value=e, name='tr_y' ) for e in y ]
        
        # loss
        loss_node = sum( [ obj.get( loss_type )( pred_node, gt_node ) * w 
                        for pred_node, gt_node, w in zip( self._out_nodes, self._gt_nodes, self._obj_weights ) ] )
        
        # gradient
        gparams = K.grad( loss_node + self._reg_value, self._params )
        
        # todo clip gradient
        if clip is not None:
            gparams = [ K.clip( gparam, -clip, clip ) for gparam in gparams ]
        
        # gradient based opt
        updates = optimizer.get_updates( self._params, gparams )
        
        # compile model
        input_nodes = self._in_nodes + self._gt_nodes
        output_nodes = [ loss_node ]
        given_nodes = sh_x + sh_y
        f = K.function_given( batch_size, input_nodes, self._tr_phase_node, output_nodes, given_nodes, updates )
        
        # debug
        # you can write debug function here
        #f_debug = K.function_no_given( self._in_nodes, self._layer_list[1].tmp )
        
        # compile for callback
        if callbacks is not None:
            callbacks = to_list( callbacks )
            for callback in callbacks:
                callback.compile( self ) 

        # train
        N = len( x[0] )
        batch_num = int( N / batch_size )
        while self._epoch < n_epoch:
            
            '''
            in_list = x+[0.]
            np.set_printoptions(threshold=np.nan, linewidth=1000, precision=10, suppress=True)
            print f_debug(*in_list)
            pause
            '''
            
            
            # callback
            for callback in callbacks:
                if ( self._epoch % callback.call_freq == 0 ):
                    callback.call()

            print
            # train
            t1 = time.time()
            for i2 in xrange(batch_num):
                loss = f(i2, 1.)[0]                     # training phase          
                if verbose: self.print_progress( self._epoch, batch_num, i2 )
            t2 = time.time()
            self._tr_time += (t2 - t1)
            self._epoch += 1
            print
            #print '\n', t2-t1, 's'          # print an empty line
        
    def predict( self, x ):
        # format data
        x = to_list( x )
        x = [ K.format_data(e) for e in x ]
        
        # compile predict model
        if not hasattr( self, '_f_predict' ):
            if len( self._out_nodes )==1:   # if only 1 out_node, then return it directly instead of list
                self._f_predict = K.function_no_given( self._in_nodes, self._tr_phase_node, self._out_nodes[0] )
            else:
                self._f_predict = K.function_no_given( self._in_nodes, self._tr_phase_node, self._out_nodes )
        
        # do predict
        in_list = x + [0.]
        y_out = self._f_predict( *in_list )
        
        return y_out
        
        
class Sequential( Model ):
    def __init__( self ):
        self._layer_seq = []
        
    def add( self, layer ):
        self._layer_seq.append( layer )
        if len( self._layer_seq ) > 1:
            self._layer_seq[-1].__call__( self._layer_seq[-2] )
            
        # regenerate model when add a new layer
        super(Sequential, self).__init__( [ self._layer_seq[0] ], [ self._layer_seq[-1] ]  )
    
class Pretrain():
    pass