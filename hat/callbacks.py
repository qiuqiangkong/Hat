'''
SUMMARY:  Callbacks, to validate, save model every several epochs
AUTHOR:   Qiuqiang Kong
Created:  2016.05.14
Modified: 2016.07.25 Modify evalute() to batch version
          2016.07.29 Fix bug in _evaluate
--------------------------------------
'''
import backend as K
import objectives as obj
import serializations
import metrics
from abc import ABCMeta, abstractmethod
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
from supports import to_list
import sys
from inspect import isfunction



'''
Callback is an abstract class
'''
class Callback( object ):
    __metaclass__ = ABCMeta
        
    @abstractmethod
    def compile( self ):
        pass
        
    @abstractmethod
    def call( self ):
        pass
        
    @property
    def call_freq( self ):
        return self._call_freq_
        
'''
Templete for creating a new Callback class
'''
class YourCallback( Callback ):
    def __init__( self, call_freq=3, # other params
                ):
        self._call_freq = call_freq
        # your code here
        
    def compile( self, md ):
        self._md = md
        # your code here
        
    def call( self, # other params
            ):
        pass
        # your code here
        
        
'''
metric_types can be list or string
'''
class Validation( Callback ):
    def __init__( self, tr_x=None, tr_y=None, va_x=None, va_y=None, te_x=None, te_y=None, batch_size=100,
                  call_freq=3, metrics=['categorical_error'], dump_path='validation.p', verbose=1):
        # init values
        self._batch_size_ = batch_size
        self._call_freq_ = call_freq
        self._metrics_ = to_list( metrics )
        self._dump_path_ = dump_path
        self._verbose_ = verbose
        self._r_ = self._init_result()
        
        # judge if train or valid or test data exists
        self._is_tr_ = self._is_data( tr_x, tr_y )
        self._is_va_ = self._is_data( va_x, va_y )
        self._is_te_ = self._is_data( te_x, te_y )
        
        # format data, to list
        if self._is_tr_:
            self._tr_x_ = [ K.format_data(e) for e in to_list( tr_x ) ]
            self._tr_y_ = [ K.format_data(e) for e in to_list( tr_y ) ]
        if self._is_va_:
            self._va_x_ = [ K.format_data(e) for e in to_list( va_x ) ]
            self._va_y_ = [ K.format_data(e) for e in to_list( va_y ) ]
        if self._is_te_:
            self._te_x_ = [ K.format_data(e) for e in to_list( te_x ) ]
            self._te_y_ = [ K.format_data(e) for e in to_list( te_y ) ]
        
    def _init_result( self ):
        r = { 'epoch': 0, 
              'cb_time': 0, 
              'tr_time': 0, 
              'call_freq': self._call_freq_, 
              }
        for metric in self._metrics_:
            metric_name = self._to_str( metric )
            r[ 'tr_'+metric_name ] = []
            r[ 'va_'+metric_name ] = []
            r[ 'te_'+metric_name ] = []
        return r
    
    def _is_data( self, x, y ):
        if ( x is not None ) and ( y is not None ):
            return True
        
    '''
    Here destroy md's closure property here for convenience. Do not use any private attributes and set method of md. 
    '''
    def compile( self, md ):
        input_nodes = md.in_nodes_
        self._f_pred = K.function_no_given( input_nodes, md.tr_phase_node_, md.out_nodes_ )
        self._md_ = md
        
        # memory usage
        print "Callback", 
        self._md_._show_memory_usage( self._md_._layer_list_, self._batch_size_ )
        
 
    def call( self ):
        t1 = time.time()
        if self._is_tr_: self._evaluate( self._tr_x_, self._tr_y_, 'tr' )
        if self._is_va_: self._evaluate( self._va_x_, self._va_y_, 'va' )
        if self._is_te_: self._evaluate( self._te_x_, self._te_y_, 'te' )
        t2 = time.time()
        self._r_['cb_time'] += t2 - t1
        self._r_['epoch'] = self._md_.epoch_
        self._r_['tr_time'] = self._md_.tr_time_
        if self._dump_path_ is not None:
            pickle.dump( self._r_, open( self._dump_path_, 'wb' ) )
        print
        
    def _evaluate( self, x, y, eval_type ):
        
        # init gt_nodes
        #gt_nodes = [ K.placeholder( e.ndim ) for e in y ]
        
        # get metric losses node
        loss_nodes = []
        for metric in self._metrics_:
            # if use default objective
            if type(metric) is str:
                assert len(self._md_.out_nodes_)==len(self._md_.gt_nodes_), "If you are using default objectives, " \
                                                + "out_node of out_layers must match ground truth!"
                loss_node = sum( [ obj.get( metric )( pred_node, gt_node ) 
                                for pred_node, gt_node in zip( self._md_.out_nodes_, self._md_.gt_nodes_ ) ] )
            # if user define their objective function
            elif isfunction( metric ):
                #loss_node = metric( self._md_.out_nodes_, self._md_.any_nodes_, gt_nodes )
                loss_node = metric( self._md_ )
            # if node
            else:
                loss_node = metric
                
            loss_nodes.append( loss_node )
        
        # compile evaluation function
        if not hasattr( self, '_f_evaluate' ):
            print 'compiling evaluation function ..'
            #input_nodes = self._md_.in_nodes_ + gt_nodes
            input_nodes = self._md_.in_nodes_ + self._md_.gt_nodes_
            self._f_evaluate = K.function_no_given( input_nodes, self._md_.tr_phase_node_, loss_nodes )
            print 'compile finished. '
        
        # calculate metric values
        t1 = time.time()
        if self._batch_size_ is None:
            in_list = x + y + [0.]
            metric_vals = np.array( self._f_evaluate( *in_list ) )
        else:
            N = len(x[0])
            batch_num = int( np.ceil( float(N) / self._batch_size_ ) )
            
            # metric_container = [ [] for e in y ]
            metric_vals = np.zeros( len(self._metrics_) )
            
            # evaluate for each batch
            for i1 in xrange( batch_num ):
                curr_batch_size = min( (i1+1)*self._batch_size_, N ) - i1*self._batch_size_
                in_list = [ e[i1*self._batch_size_ : min( (i1+1)*self._batch_size_, N ) ] for e in x ] \
                        + [ e[i1*self._batch_size_ : min( (i1+1)*self._batch_size_, N ) ] for e in y ] + [0.]
                batch_metric_vals = np.array( self._f_evaluate( *in_list ) )
                metric_vals += batch_metric_vals * curr_batch_size
                if self._verbose_==1: self._print_progress( batch_num, i1 )
                
            metric_vals /= N
            
        # timer
        t2 = time.time()
            
        # print results
        self._print_time_results( eval_type, self._metrics_, metric_vals, t2-t1 )
            
                
    # print progress on screen
    def _print_progress( self, batch_num, curr_batch_num ):
        sys.stdout.write("testing: %d%%   \r" % ( float(curr_batch_num)/float(batch_num)*100 ) )
        sys.stdout.flush()
        
    def _print_time_results( self, eval_type, metrics, metric_vals, time ):
        chs = "    "
        for i1 in xrange( len(metrics) ):
            metric_name = self._to_str( metrics[i1] )
            chs += eval_type + "_" + metric_name + ": %.5f" % metric_vals[i1] + "    | "
        chs += "time: %.2f" % time
        print chs
        
    def _to_str( self, a ):
        if type(a) is str:
            return a
        elif isfunction( a ):
            return a.__name__
        else:
            return 'reg_value'
        
'''
Save Model every n-epoches
'''
class SaveModel( Callback ):
    def __init__( self, dump_fd, call_freq=3 ):
        self._dump_fd_ = dump_fd
        self._call_freq_ = call_freq
        
    def compile( self, md ):
        self._md_ = md
        
    def call( self ):
        dump_path = self._dump_fd_ + '/md' + str(self._md_.epoch_) + '.p'
        serializations.save( self._md_, dump_path )
        
            
        
class Debug( Callback ):
    def __init__( self, call_freq, x,# other params
                ):
        self._call_freq = call_freq
        x = K.format_data(x)
        self._x = [x]
        self._counter = 0
        
    def compile( self, md ):
        self._md = md
        self._f = K.function_no_given( md.in_nodes, md.tr_phase_node, md._layer_seq[3].output )
        
    def call( self ):
        in_list = self._x + [0.]
        
        np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2, suppress=True)
        self._counter += 1
        #print self._f( *in_list )
        if self._counter==3:
            print 'asdf'
            print self._f(*in_list)
            pause
    
