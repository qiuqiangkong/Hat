'''
SUMMARY:  Callbacks, to validate, save model every several epochs
AUTHOR:   Qiuqiang Kong
Created:  2016.05.14
Modified: -
--------------------------------------
'''
import backend as K
import objectives as obj
import metrics
from abc import ABCMeta, abstractmethod
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
from supports import to_list



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
        return self._call_freq
        
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
    def __init__( self, tr_x=None, tr_y=None, va_x=None, va_y=None, te_x=None, te_y=None, 
                  call_freq=3, metric_types=['categorical_error'], dump_path='validation.p'):
        # init values
        self._call_freq = call_freq
        self._metric_types = to_list( metric_types )
        self._dump_path = dump_path
        self._r = self._init_result()
        
        # judge if train or valid or test data exists
        self._is_tr = self._isData( tr_x, tr_y )
        self._is_va = self._isData( va_x, va_y )
        self._is_te = self._isData( te_x, te_y )
        
        # format data, to list
        if self._is_tr:
            self._tr_x = [ K.format_data(e) for e in to_list( tr_x ) ]
            self._tr_y = [ K.format_data(e) for e in to_list( tr_y ) ]
        if self._is_va:
            self._va_x = [ K.format_data(e) for e in to_list( va_x ) ]
            self._va_y = [ K.format_data(e) for e in to_list( va_y ) ]
        if self._is_te:
            self._te_x = [ K.format_data(e) for e in to_list( te_x ) ]
            self._te_y = [ K.format_data(e) for e in to_list( te_y ) ]
        
    def _init_result( self ):
        r = { 'epoch': 0, 
              'cb_time': 0, 
              'tr_time': 0, 
              'call_freq': self._call_freq, 
              }
        for metric in self._metric_types:
            r[ 'tr_'+metric ] = []
            r[ 'va_'+metric ] = []
            r[ 'te_'+metric ] = []
        return r
    
    def _isData( self, x, y ):
        if ( x is not None ) and ( y is not None ):
            return True
        
    '''
    Here destroy md's closure property here for convenience. Do not use any private attributes and set method of md. 
    '''
    def compile( self, md ):
        input_nodes = md.in_nodes
        self._f_pred = K.function_no_given( input_nodes, md.tr_phase_node, md.out_nodes )
        self._md = md
        
 
    def call( self ):
        t1 = time.time()
        if self._is_tr: self._evaluate( self._tr_x, self._tr_y, 'tr' )
        if self._is_va: self._evaluate( self._va_x, self._va_y, 'va' )
        if self._is_te: self._evaluate( self._te_x, self._te_y, 'te' )
        t2 = time.time()
        self._r['cb_time'] += t2 - t1
        self._r['epoch'] = self._md.epoch
        self._r['tr_time'] = self._md.tr_time
        pickle.dump( self._r, open( self._dump_path, 'wb' ) )
        print
        
    def _evaluate( self, x, y, test_type ):
        in_list = x + [0.]
        n_out_nodes = len(y)
        
        for metric in self._metric_types:
            y_out = self._f_pred( *in_list )
            val = 0.
            
            
            if metric=='prec_recall_fvalue':
                for i1 in xrange( n_out_nodes ):
                    prec, recall, fvalue = metrics.get( metric )( y_out[i1], y[i1], 0.5 )
                    val += fvalue
            else:
                for i1 in xrange( n_out_nodes ):
                    val += metrics.get( metric )( y_out[i1], y[i1] )
                
            key = test_type + '_' + metric
            self._r[ key ].append( val )
            if metric!='confusion_matrix':
                print key, val, 

'''
Save Model every n-epoches
'''
class SaveModel( Callback ):
    def __init__( self, dump_fd, call_freq=3 ):
        self._dump_fd = dump_fd
        self._call_freq = call_freq
        
    def compile( self, md ):
        self._md = md
        
    def call( self ):
        dump_path = self._dump_fd + '/md' + str(self._md.epoch) + '.p'
        try:
            pickle.dump( self._md, open( dump_path, 'wb' ) )
        except:
            assert False, "Did you create a folder named '" + self._dump_fd + "'?"
            
        
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
    
