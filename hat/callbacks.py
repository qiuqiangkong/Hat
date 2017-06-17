"""
SUMMARY:  Callbacks can be executed after each training epoch. 
AUTHOR:   Qiuqiang Kong
Created:  2016.05.14
--------------------------------------
"""
import backend as K
import objectives as obj
from generators import BaseGenerator
import serializations
import metrics
from abc import ABCMeta, abstractmethod
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
from supports import to_list, format_data_list
import sys
from inspect import isfunction



class Callback(object):
    """
    Callback is an abstract class
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, call_freq, type):
        assert (type=='epoch' or type=='iter'), "Error! type must be 'epoch' | 'iter'!"
        self._call_freq_ = call_freq
        self._type_ = type
        
    @abstractmethod
    def compile(self):
        pass
        
    @abstractmethod
    def call(self):
        pass
        
    @property
    def call_freq_(self):
        return self._call_freq_
        
    @property
    def type_(self):
        return self._type_
        

class YourCallback(Callback):
    """
    Templete for users to create a new Callback class
    """
    def __init__(self, call_freq=3, type='epoch', # other params
               ):
        super(YourCallback, self).__init__(call_freq, type)
        # your code here
        
    def compile(self, md):
        self._md = md_
        # your code here
        
    def call(self, # other params
           ):
        pass
        # your code here
        
        
class SaveModel(Callback):
    """
    Save Model every n-epoches
    """
    def __init__(self, dump_fd, call_freq=3, type='epoch'):
        super(SaveModel, self).__init__(call_freq, type)
        self._dump_fd_ = dump_fd
        
    def compile(self, md):
        self._md_ = md
        
    def call(self):
        if self.type_ == 'epoch':
            dump_path = self._dump_fd_ + '/md' + str(self._md_.epoch_) + '_epochs.p'
        elif self.type_ == 'iter':
            dump_path = self._dump_fd_ + '/md' + str(self._md_.iter_) + '_iters.p'
            
        serializations.save(self._md_, dump_path)
        print "    Save to " + dump_path + " successfully!\n"
        


class Validation(Callback):
    """Validate existing model using train, validation and test data. 
    
    Args:
      tr_x: ndarray | list of ndarray | None
      tr_y: ndarray | list of ndarray | None
      va_x: ndarray | list of ndarray | None
      va_y: ndarray | list of ndarray | None
      va_x: ndarray | list of ndarray | None
      va_y: ndarray | list of ndarray | None
      batch_size: integar
      call_freq: integar. Validation is called every #call_freq epoches or iters. 
      metrics: string | function. 
      dump_path: string | None
      type: 'epoch' | 'iter'
    """
    def __init__(self, tr_x=None, tr_y=None, va_x=None, va_y=None, 
                 te_x=None, te_y=None, batch_size=100, call_freq=3, 
                 metrics=['categorical_error'], generator=None, transformer=None, 
                 dump_path=None, type='epoch', verbose=1):
        # init values
        super(Validation, self).__init__(call_freq, type)
        self._batch_size_ = batch_size
        self._metrics_ = to_list(metrics)
        self._generator_ = generator
        self._transformer_ = transformer
        self._dump_path_ = dump_path
        self._type_ = type
        self._verbose_ = verbose
        self._r_ = self._init_result()
        
        # judge if train or valid or test data exists
        self._is_tr_ = self._is_data(tr_x, tr_y)
        self._is_va_ = self._is_data(va_x, va_y)
        self._is_te_ = self._is_data(te_x, te_y)
        
        self._tr_x_ = tr_x
        self._tr_y_ = tr_y
        self._va_x_ = va_x
        self._va_y_ = va_y
        self._te_x_ = te_x
        self._te_y_ = te_y
        
    def _init_result(self):
        r = {'epoch': 0, 
              'cb_time': 0, 
              'tr_time': 0, 
              'call_freq': self._call_freq_, 
             }
        for metric in self._metrics_:
            metric_name = self._to_str(metric)
            r['tr_'+metric_name] = []
            r['va_'+metric_name] = []
            r['te_'+metric_name] = []
        return r
    
    def _is_data(self, x, y):
        if (x is not None) or (y is not None):
            return True
        else:
            return False
        
    """
    Here destroy md's closure property for convenience. Do not do any changes to md. 
    """
    def compile(self, md):
        inputs = md.in_nodes_ + [md.tr_phase_node_]
        self._f_pred = K.function_no_given(inputs, md.out_nodes_)
        self._md_ = md
        
        # memory usage
        print "Callback", 
        self._md_._show_memory_usage(self._md_.effective_layers_, self._batch_size_)
        
 
    def call(self):
        t1 = time.time()
        if self._is_tr_: self._evaluate(self._tr_x_, self._tr_y_, 'tr')
        if self._is_va_: self._evaluate(self._va_x_, self._va_y_, 'va')
        if self._is_te_: self._evaluate(self._te_x_, self._te_y_, 'te')
        t2 = time.time()
        self._r_['cb_time'] += t2 - t1
        self._r_['epoch'] = self._md_.epoch_
        self._r_['tr_time'] = self._md_.tr_time_
        if self._dump_path_ is not None:
            pickle.dump(self._r_, open(self._dump_path_, 'wb'))
        print
        
    def _evaluate(self, x, y, eval_type):
        # get metric losses node
        loss_nodes = []
        for metric in self._metrics_:
            # if use default objective
            if type(metric) is str:
                assert len(self._md_.out_nodes_)==len(self._md_.gt_nodes_), "If you are using default objectives, " \
                                                + "out_node of out_layers must match ground truth!"
                loss_node = sum([obj.get(metric)(pred_node, gt_node) 
                                for pred_node, gt_node in zip(self._md_.out_nodes_, self._md_.gt_nodes_)])
            # if user define their objective function
            elif isfunction(metric):
                loss_node = metric(self._md_)
            else:
                loss_node = metric
                
            loss_nodes.append(loss_node)
        
        # compile evaluation function
        if not hasattr(self, '_f_evaluate'):
            print 'compiling evaluation function ..'
            inputs = self._md_.in_nodes_ + self._md_.gt_nodes_ + [self._md_.tr_phase_node_]
            self._f_evaluate = K.function_no_given(inputs, loss_nodes)
            print 'compile finished. '
        
        # calculate metric values
        t1 = time.time()
        if self._generator_:
            generator = self._generator_
        else:
            generator = self._DefaultGenerator(self._batch_size_)
        n_all = 0.
        cnt= 0.
        metric_vals = np.zeros(len(self._metrics_))
        batch_num = sum(1 for it in generator.generate(x, y))
        for batch_x, batch_y in generator.generate(x, y):
            batch_x = to_list(batch_x)
            batch_y = to_list(batch_y)
            curr_batch_size = batch_x[0].shape[0]
            n_all += curr_batch_size
            cnt += 1.
            if self._transformer_:
                (batch_x, batch_y) = self._transformer_.transform(batch_x, batch_y)
            batch_x = format_data_list(batch_x)
            batch_y = format_data_list(batch_y)
            in_list = batch_x + batch_y + [0.]
            batch_metric_vals = np.array(self._f_evaluate(*in_list))
            metric_vals += batch_metric_vals * curr_batch_size
            if self._verbose_==1: self._print_progress(batch_num, cnt)
        metric_vals /= n_all
            
        # timer
        t2 = time.time()
            
        # print results
        self._print_time_results(eval_type, self._metrics_, metric_vals, t2-t1)
            
    class _DefaultGenerator(BaseGenerator):
        def __init__(self, batch_size):
            self._batch_size_ = batch_size
        
        def generate(self, x, y):
            x = to_list(x)
            y = to_list(y)
            N = len(x[0])
            batch_num = int(np.ceil(float(N) / self._batch_size_))
            
            # evaluate for each batch
            for i1 in xrange(batch_num):
                curr_batch_size = min((i1+1)*self._batch_size_, N) - i1*self._batch_size_
                batch_x = [e[i1*self._batch_size_ : min((i1+1)*self._batch_size_, N)] for e in x]
                batch_y = [e[i1*self._batch_size_ : min((i1+1)*self._batch_size_, N)] for e in y]
                yield batch_x, batch_y
            
    # print progress on screen
    def _print_progress(self, batch_num, curr_batch_num):
        sys.stdout.write("testing: %d%%   \r" % (float(curr_batch_num)/float(batch_num)*100))
        sys.stdout.flush()
        
    def _print_time_results(self, eval_type, metrics, metric_vals, time):
        chs = "    "
        for i1 in xrange(len(metrics)):
            metric_name = self._to_str(metrics[i1])
            chs += eval_type + "_" + metric_name + ": %.5f" % metric_vals[i1] + "    | "
        chs += "time: %.2f" % time
        print chs
        
    def _to_str(self, a):
        if type(a) is str:
            return a
        elif isfunction(a):
            return a.__name__
        else:
            return 'reg_value'
        
        