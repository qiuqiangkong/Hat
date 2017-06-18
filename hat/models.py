'''
SUMMARY:  Build Model
AUTHOR:   Qiuqiang Kong
Created:  2016.05.01
Modified: 2017.02.19
--------------------------------------
'''

import sys
from supports import to_list, shuffle, format_data_list, memory_usage, Timer
import numpy as np
import supports
from optimizers import *
from globals import reset_id_to_zero
import activations
import objectives as obj
import backend as K
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle


class Model(object):
    """Build deep learning model.
    Methods:
      summary()
      get_effective_layers()
      set_layers_trainability()
      set_trainability()
      find_layer()
      add_models()
      joint_models()
      compile()
      get_optimization_func()
      fit()
      train_on_batch()
      predict()
      get_observe_forward_func()
      get_observe_backward_func()
      run_function()
      
    Args:
      in_layers: a list of layer
      out_layers: a list of layer
      any_layers: a list of layer
    """
    def __init__(self, in_layers=[], out_layers=[], any_layers=[]):
        self._in_layers_ = to_list(in_layers)
        self._out_layers_ = to_list(out_layers)
        self._any_layers_ = to_list(any_layers)

        self._gt_nodes_ = None
        self._f_predict = None
        
        self._epoch_ = 0
        self._iter_= 0
        self._tr_time_ = 0.
        
        self._tr_phase_node_ = K.common_tr_phase_node
        self._effective_layers_ = self.get_effective_layers(self._in_layers_, self._out_layers_)
        
        self._check_duplicate_name(self._effective_layers_)
        self._trainable_table_ = self._init_trainable_table(self._effective_layers_)
        
        self._f_optimize_ = None

    # ------------------ Private methods ------------------
    
    def _depth_first_search(self, layer, visited, type):
        """
        Depth first search (DFS) of a given layer. 
        
        Args:
          layer: layer. From this layer to start depth first search (DFS). 
          visited: list of layer. Layers already visited and to be skipped in DFS. 
          type: 'forward' | 'backward'. 'forward' is doing BFS by layer.nexts_, 
              'backward' is doing BFS by layer.prevs_
              
        Return:
          list of layers. Add layers by DFS to existing visited layers. 
        """
        visited.append(layer)
        son_layers = self._get_son_layers(layer, type)
        for son_layer in son_layers:
            if son_layer not in visited:
                self._depth_first_search(son_layer, visited, type)
        return visited
        
    def _get_son_layers(self, layer, type):
        """Get son layers of a layer, type can be 'forward' | 'backward'
        """
        if type=='forward':
            return layer.nexts_
        elif type=='backward':
            return layer.prevs_
        else:
            raise Exception("type should be 'forward' | 'backward'!")

    def _check_duplicate_name(self, effective_layers):
        """Check if duplicated name in effective layers. 
        """
        from collections import Counter
        names = [layer.name_ for layer in effective_layers]
        duplicated_names = [name for name,v in Counter(names).items() if v>1]
        if duplicated_names:
            err_str = "Layers must have unique names! Names "
            for name in duplicated_names: err_str = err_str + "'" + name + "' "
            err_str += "are not unique!" 
            raise Exception(err_str)

    def _init_trainable_table(self, effective_layers):
        """Trainable_table has [index, layer, bool]. 
        """
        trainable_table = []
        for layer in effective_layers:
            index = len(trainable_table)
            trainable_table.append([index, layer, True])
        return trainable_table
        
    def _get_all_params(self, effective_layers, trainable_table):
        """Return list of all trainable layer's params. 
        """
        params = []
        for row in trainable_table:
            [index, layer, trainable] = row
            if trainable is True:
                if hasattr(layer, 'params_'):
                    params += layer.params_
        return params
        
    def _get_all_inner_updates(self, effective_layers, trainable_table):
        """Return list of all trainable layer's inner_updates. 
        """
        inner_updates = []
        for row in trainable_table:
            [index, layer, trainable] = row
            if trainable is True:
                if hasattr(layer, 'inner_updates_'):
                    inner_updates += layer.inner_updates_
        return inner_updates
        
    def _get_all_layer_reg_value(self, effective_layers, trainable_table):
        """Return list of all trainable layer's reg_value. 
        """
        reg_value = 0.
        for row in trainable_table:
            [index, layer, trainable] = row
            if trainable is True:
                if hasattr(layer, 'reg_value_'):
                    reg_value += layer.reg_value_
        return reg_value
        
    def _find_tt_row_by_name(self, layer_name, trainable_table):
        for row in trainable_table:
            if row[1].name_ == layer_name:
                return row
        return None
        
    def _show_memory_usage(self, effective_layers, batch_size):
        n_data_ary = []
        for layer in effective_layers:
            n_data_ary.append(np.prod(layer.out_shape_[1:]) * batch_size)
        
        n_data_total = np.sum(n_data_ary)
        n_byte_total = n_data_total * 4   # float32 
        n_byte_total *= 2                 # forward & backward
        f = "{0:<30} {1:<10}"
        print f.format("total memory usage:", str(n_byte_total/1e6)+"Mb")
        
    def _print_progress(self, epoch, batch_num, curr_batch_num):
        sys.stdout.write("%d-th epoch %d%% \r" % (epoch, float(curr_batch_num)/float(batch_num)*100))
        sys.stdout.flush()
    
    def _print_progress_loss(self, epoch, batch_num, curr_batch_num, loss):
        sys.stdout.write("%d-th epoch %d%%   %s %f \r" % (epoch, float(curr_batch_num)/float(batch_num)*100, 'tr_loss:', loss))
        sys.stdout.flush()
        
    def _n_layer_params(self, layer):
        n_params = 0
        for param in layer.params_:
            n_params += np.prod(K.get_value(param).shape)
        return n_params
        
    def _n_model_trainable_params(self):
        n_params = 0
        for row in self._trainable_table_:
            [index, layer, bool] = row
            if bool is True:
                n_params += self._n_layer_params(layer)
        return n_params
        
    def set_gt_nodes(self, target_dim_list):
        """Set gt_nodes for computing loss for training. 
        """
        self._gt_nodes_ = [K.placeholder(y_dim) for y_dim in target_dim_list]
        
    def _set_epoch(self, epoch):
        self._epoch_ = epoch
        
    def _set_iter(self, iter):
        self._iter_ = iter
        
    # get target dim list if transformer exists or not
    def _get_target_dim_list(self, x, y, transformer):
        if not transformer:
            target_dim_list = [e.ndim for e in y]
        else:
            test_x = [e[0:2] for e in x]
            test_y = [e[0:2] for e in y]
            (_, test_y) = transformer.transform(test_x, test_y)
            target_dim_list = [e.ndim for e in test_y]
        return target_dim_list
    
    # ------------------ Public methods ------------------
    
    def summary(self, verbose=1):
        """Print model information. 
        """
        def _get_params_str(params):
            string = ""
            for p in params:
                string = string + p.name.split('_')[-1] + ", "
            return string[:-2]
        
        if verbose==1:
            print '---------- summary -----------'
            f = "{0:<20} {1:<10} {2:<25} {3:<20} {4:<15} {5:<20} {6:<20}"
            print f.format('layer_name', 'layer_id', 'prev layers', 'trainable_params', 'n_params', 'out_shape', 'trainable')
            for row in self.trainable_table_:
                [index, layer, trainable] = row
                prev_layer_names = [prev_layer.name_ for prev_layer in layer.prevs_]
                n_params = self._n_layer_params(layer)
                print f.format(layer.name_, layer.id_, prev_layer_names, 
                            _get_params_str(layer.params_), n_params, str(layer.out_shape_), trainable)
            print "\nTotal trainable params:", self._n_model_trainable_params(), "\n"
        
        elif verbose==2:
            print '---------- summary -----------'
            f = "{0:<10} {1:<20} {2:<10} {3:<25} {4:<20} {5:<15} {6:<20} {7:<20}"
            print f.format('tb_index', 'layer_name', 'layer_id', 'prev layers', 'trainable_params', 'n_params', 'out_shape', 'trainable')
            for row in self.trainable_table_:
                [index, layer, trainable] = row
                prev_layer_names = [prev_layer.name_ for prev_layer in layer.prevs_]
                n_params = self._n_layer_params(layer)
                print f.format(index, layer.name_, layer.id_, prev_layer_names, 
                            _get_params_str(layer.params_), n_params, str(layer.out_shape_), trainable)
            print "\nTotal trainable params:", self._n_model_trainable_params(), "\n"
        
    def get_effective_layers(self, in_layers, out_layers):
        """Return intersection of forward_visited and backward_visited layers. 
        """
        # DFS of in_layers
        forward_visited = []
        for in_layer in in_layers:
            forward_visited = self._depth_first_search(in_layer, forward_visited, type='forward')
            
        # DFS of out_layers
        backward_visited = []
        for out_layer in out_layers:
            backward_visited = self._depth_first_search(out_layer, backward_visited, type='backward')
            
        # Intersection of forward_visited and backward_visited
        effective_layers = [layer for layer in forward_visited if layer in backward_visited]
        
        # Sort effective_layers by id
        effective_layers = sorted(effective_layers, key=lambda e: e.id_, reverse=False)
        
        return effective_layers
   
    def set_layers_trainability(self, layer_names, bool):
        """Set list of layers' trainablility. 
        """
        layer_names = to_list(layer_names)

        for layer_name in layer_names:
            row = self._find_tt_row_by_name(layer_name, self._trainable_table_)
            if row is None: 
                raise Exception("Can not find layer_name!")
            else: 
                index = row[0]
            self._trainable_table_[index][-1] = bool
        
        # self._refresh_model()
        
    def set_trainability(self, bool):
        """Set all the trainability of all layers in trainable_table. 
        """
        for row in self._trainable_table_:
            row[-1] = bool
            
        # self._refresh_model()

    def find_layer(self, name):
        """Search and return a layer in the model by name. 
        """
        for layer in self._effective_layers_:
            if name==layer.name_:
                return layer
        raise Exception("No layer named " + name + "! ")
        
    def add_models(self, mds):
        """Add list of model to existing model, update in_layers, out_layers, any_layers and trainable_table. 
        """
        mds = to_list(mds)
        
        new_in_layers = self.in_layers_
        new_out_layers = self.out_layers_
        new_any_layers = self.any_layers_
        init_trainable_table = self.trainable_table_
        for md in mds:
            new_in_layers += md.in_layers_
            new_out_layers += md.out_layers_
            new_any_layers += md.any_layers_
            init_trainable_table += md.trainable_table_
        
        self.__init__(new_in_layers, new_out_layers, new_any_layers)
        
        # New trainable_table is inherited from old concatenated trainable_table
        for row1 in init_trainable_table:
            row2 = self._find_tt_row_by_name(row1[1].name_, self._trainable_table_)
            if row2:
                index = row2[0]
                self._trainable_table_[index][-1] = row1[-1]

    def joint_models(self, tail_layer_name, head_layer_name):
        """Joint tail node and head node, update out_layers and in_layers. 
        """
        tail_layer = self.find_layer(tail_layer_name)
        head_layer = self.find_layer(head_layer_name)
        
        # Joint
        tail_layer.add_next(head_layer)
        head_layer.set_previous(tail_layer)
        
        # After joint, the tail & head layer will be removed from out_layers & in_layers
        def _remove_layer_from_list(name, layer_list):
            for layer in layer_list:
                if name==layer.name_:
                    layer_list.remove(layer)
                    return layer_list
        
        self._in_layers_ = _remove_layer_from_list(head_layer.name_, self._in_layers_)
        self._out_layers_ = _remove_layer_from_list(tail_layer.name_, self._out_layers_)
        
        # self._refresh_model()
        
    def compile(self):
        """Compile the computation graph, update params, reg_value, inner_updates. 
        """
        for layer in self._effective_layers_:
            layer.compile()
            
    def get_optimization_func(self, target_dim_list, loss_func, optimizer, clip):
        """Compile and return optimization function. 
        
        Args:
          target_dim_list: list of integars. targets' dimension. e.g. target_dim_list=[2]
          loss_func: string | function. 
          optimizer: object. 
          clip: None | real value. 
          
        Return:
          optimization function. 
        """
        # set gt nodes
        self.set_gt_nodes(target_dim_list)
        
        # Default loss
        if type(loss_func) is str:
            assert len(self.out_nodes_)==1, "If the number of out_layers > 1, you need define your own loss_func!"
            loss_node = obj.get(loss_func)(self.out_nodes_[0], self.gt_nodes_[0])
            
        # User defined loss
        else:
            loss_node = loss_func(self)
         
        # Compute gradient
        gparams = K.grad(loss_node + self.reg_value_, self.params_)
        
        # Clip gradient
        if clip is not None:
            gparams = [K.clip(gparam, -clip, clip) for gparam in gparams]
        
        # Gradient based optimization
        param_updates = optimizer.get_updates(self.params_, gparams)
        
        # Get all updates
        updates = param_updates + self.inner_updates_
        
        # Compile model
        inputs = self.in_nodes_ + self.gt_nodes_ + [K.common_tr_phase_node]
        outputs = [loss_node]
        f = K.function_no_given(inputs, outputs, updates)
        return f
     
    def fit(self, x, y, batch_size=100, n_epochs=10, loss_func='categorical_crossentropy', 
             optimizer=SGD(lr=0.01, momentum=0.9), clip=None, callbacks=[], shuffle=True, transformer=None, verbose=1):
        """Fit data and train model. The data is reshuffled after every epoch. 
        
        Args:
          x: ndarray | list of numpy ndarray. 
          y: ndarray | list of numpy ndarray. 
          batch_size: integar. 
          n_epoch: integar. Number of training epochs. 
          loss_func: str | function. 
          optimizer: optimization object. 
          clip: real value. 
          callbacks: list of Callback object. 
          shuffle: bool. 
          transformer: Transformation object. 
          verbose: 0 | 1 | 2
          
        Returns: None. (All trained models, results should be saved using callbacks.)
        """
        x = to_list(x)
        y = to_list(y)
        
        # Format data
        x = format_data_list(x)
        y = format_data_list(y)
        
        # Train memory usage
        print "Training", 
        self._show_memory_usage(self._effective_layers_, batch_size)
        
        # Compile optimization function
        timer = Timer()
        target_dim_list = self._get_target_dim_list(x, y, transformer)
        f_optimize = self.get_optimization_func(target_dim_list, loss_func, optimizer, clip)
        timer.show("Compiling f_optimize time:")
        
        # Compile for callback
        timer = Timer()
        if callbacks is not None:
            callbacks = to_list(callbacks)
            for callback in callbacks:
                callback.compile(self) 
        timer.show("Compiling callbacks time:")
        
        # Train
        N = len(x[0])
        batch_num = int(np.ceil(float(N) / batch_size))
        max_epoch = n_epochs + self.epoch_
        
        # Callback
        print '\n', self.epoch_, 'th epoch:'
        for callback in callbacks:
            if (self.epoch_ % callback.call_freq_ == 0):
                callback.call()
        
        while self.epoch_ < max_epoch:
            # Shuffle data
            if shuffle: x, y = supports.shuffle(x, y)

            # Train
            t1 = time.time()
            loss_list = []
            for i2 in xrange(batch_num):
                batch_x = [e[i2*batch_size : min((i2+1)*batch_size, N)] for e in x]
                batch_y = [e[i2*batch_size : min((i2+1)*batch_size, N)] for e in y]
                if transformer: 
                    (batch_x, batch_y) = transformer.transform(batch_x, batch_y)
                    batch_x = format_data_list(batch_x)
                    batch_y = format_data_list(batch_y)
                in_list = batch_x + batch_y + [1.]      # training phase
                loss = f_optimize(*in_list)[0]                      
                loss_list.append(loss)
                # self._iter_ += 1
                if verbose==1: self._print_progress(self.epoch_, batch_num, i2)
                if verbose==2: self._print_progress_loss(self.epoch_, batch_num, i2, loss)
                
            t2 = time.time()
            self._tr_time_ += (t2 - t1)            
            if verbose!=0: print '\n', '    tr_time: ', "%.2f" % (t2-t1), 's'          # print an empty line
            self._epoch_ += 1
            
            # Callback
            for callback in callbacks:
                if (self.epoch_ % callback.call_freq_ == 0):
                    callback.call()
    
    def train_on_batch(self, batch_x, batch_y, loss_func='categorical_crossentropy', 
                       optimizer=SGD(lr=0.01, momentum=0.9), clip=None, callbacks=[], 
                       transformer=None, recompile=False):
        """Train model on single batch data. 
        
        Args:
          batch_x: ndarray | list of ndarray. 
          batch_y: ndarray | list of ndarray. 
          loss_func: str | function. 
          optimizer: optimization object. 
          clip: real value. 
          callbacks: list of Callback object. 
          transformer: Transformation object. 
          recompile: bool. Recompile the optimize function or not. 
          
        Returns: None. (All trained models, results should be saved using callbacks.)
        """
        batch_x = to_list(batch_x)
        batch_y = to_list(batch_y)
        
        # Format data
        batch_x = format_data_list(batch_x)
        batch_y = format_data_list(batch_y)
        
        # Compile optimization function
        if (not self._f_optimize_) or (recompile):
            timer = Timer()
            target_dim_list = self._get_target_dim_list(batch_x, batch_y, transformer)
            self._f_optimize_ = self.get_optimization_func(target_dim_list, loss_func, optimizer, clip)
            timer.show("Compiling f_optimize time:")
            
            # Compile for callback
            timer = Timer()
            if callbacks is not None:
                callbacks = to_list(callbacks)
                for callback in callbacks:
                    callback.compile(self) 
            timer.show("Compiling callbacks time:")
        
        
        for callback in callbacks:
            if (self.iter_ % callback.call_freq_ == 0):
                print self.iter_, 'th iteration:'
                callback.call()
        
        # Train
        t1 = time.time()
        if transformer: 
            (batch_x, batch_y) = transformer.transform(batch_x, batch_y)
            batch_x = format_data_list(batch_x)
            batch_y = format_data_list(batch_y)
        in_list = batch_x + batch_y + [1.]      # training phase
        loss = self._f_optimize_(*in_list)[0]
        self._iter_ += 1
        t2 = time.time()
        self._tr_time_ += (t2 - t1)
        return loss
        
    def train_on_batch_with_func(self, func, batch_x, batch_y):
        """Train model on batch data. 
        
        Args:
          func: function. 
          batch_x: ndarray | list of ndarray. 
          batch_y: ndarray | list of ndarray. 
        """
        batch_x = to_list(batch_x)
        batch_y = to_list(batch_y)
        
        # Format data
        batch_x = format_data_list(batch_x)
        batch_y = format_data_list(batch_y)
        
        in_list = batch_x + batch_y + [1.]      # training phase
        loss = func(*in_list)[0]   
        
        self._set_iter(self.iter_ + 1)                   
        return loss
        
    def fit_generator(self, x, y, generator, loss_func='categorical_crossentropy', 
             optimizer=SGD(lr=0.01, momentum=0.9), clip=None, callbacks=[], transformer=None, verbose=1):
        
        # Train
        for batch_x, batch_y in generator.generate(x, y):
            batch_x = to_list(batch_x)
            batch_y = to_list(batch_y)
            t1 = time.time()
            
            # Compile optimization function
            if not self._f_optimize_:
                # Train memory usage
                batch_size = len(batch_x[0])
                print "Training", 
                self._show_memory_usage(self._effective_layers_, batch_size)
                
                timer = Timer()
                target_dim_list = self._get_target_dim_list(batch_x, batch_y, transformer)
                self._f_optimize_ = self.get_optimization_func(target_dim_list, loss_func, optimizer, clip)
                timer.show("Compiling f_optimize time:")
                
                # Compile for callback
                timer = Timer()
                if callbacks is not None:
                    callbacks = to_list(callbacks)
                    for callback in callbacks:
                        callback.compile(self) 
                timer.show("Compiling callbacks time:")
            
            if transformer: 
                (batch_x, batch_y) = transformer.transform(batch_x, batch_y)
            batch_x = format_data_list(batch_x)
            batch_y = format_data_list(batch_y)
            
            # Callback
            for callback in callbacks:
                if (self.iter_ % callback.call_freq_ == 0):
                    print
                    callback.call()
            
            in_list = batch_x + batch_y + [1.]      # training phase
            loss = self._f_optimize_(*in_list)[0]
            self._iter_ += 1
            
            t2 = time.time()
            self._tr_time_ += (t2 - t1)
            sys.stdout.write("iteration: %d  loss: %f  time per batch: %.2f \r" % (self._iter_, loss, t2-t1))
            sys.stdout.flush()
        
    def predict(self, x, batch_size=128, tr_phase=0.):
        """Predict output using current model. 
        
        Args:
          x: ndarray | list of ndarray. 
          batch_size: integar | None. Predict batch size. 
          tr_phase: 0. | 1. Test phase or train phase. 
          
        Returns:
          ndarray | list of ndarray. 
        """
        # Compile predict model just once
        if self._f_predict is None:
            inputs = self.in_nodes_ + [self.tr_phase_node_]
            timer = Timer()
            self._f_predict = K.function_no_given(inputs, self.out_nodes_)
            timer.show("Compiling f_predict time:")
        
        return self.run_function(self._f_predict, x, batch_size, tr_phase)
        
    def get_observe_forward_func(self, observe_nodes):
        observe_nodes = to_list(observe_nodes)
        inputs = self.in_nodes_ + [self.tr_phase_node_]
        timer = Timer()
        f_observe_forward = K.function_no_given(inputs, observe_nodes)
        timer.show("Compiling f_observe_forward time:")
        return f_observe_forward
        
    def get_observe_backward_func(self, observe_nodes):
        if self.gt_nodes_ is None:
            raise Exception("You must call set_gt_nodes method before call observe_backward method!")
            
        observe_nodes = to_list(observe_nodes)
        inputs = self.in_nodes_ + self.gt_nodes_ + [self.tr_phase_node_]
        timer = Timer()
        f_observe_backward = K.function_no_given(inputs, observe_nodes)
        timer.show("Compiling f_observe_backward time:")
        return f_observe_backward
        
    def run_function(self, func, z, batch_size, tr_phase):
        """Return output of a function given value. 
        
        Args:
        func: function
        z: ndarray | list of ndarray. Can be [inputs] for computing forward, 
            or [inputs]+[outputs] for computing backward
            
        Returns:
        list of ndarray
        """
        # Format data
        z = to_list(z)
        z = format_data_list(z)
        
        # Calculating all in same time
        if batch_size is None:
            in_list = z + [tr_phase]
            y_out = func(*in_list)
        # Calculating in batch
        else:
            N = len(z[0])
            batch_num = int(np.ceil(float(N) / batch_size))
            n_out_nodes = len(self.out_nodes_)
            y_out = []      # list of batch_y_out
            for i1 in xrange(batch_num):
                in_list = [e[i1*batch_size : min((i1+1)*batch_size, N)] for e in z] + [tr_phase]
                batch_y_out = func(*in_list)    # list of ndarray
                y_out.append(batch_y_out)

            def _reform(y_out):
                outs = []
                for i1 in xrange(len(y_out[0])):
                    tmp_list = []
                    for j1 in xrange(len(y_out)):
                        tmp_list.append(y_out[j1][i1])
                    out = np.concatenate(tmp_list, axis=0)
                    outs.append(out)
                return outs
                    
            y_out = _reform(y_out)
        
        return y_out
            
    @property
    def info_( self ):
        dict = { 'epoch': self.epoch_, 
                 'iter': self.iter_, 
                 'in_ids': [ layer.id_ for layer in self._in_layers_ ], 
                 'out_ids': [ layer.id_ for layer in self._out_layers_ ], 
                 'any_ids': [layer.id_ for layer in self._any_layers_] }
        return dict
    
    @classmethod
    def load_from_info( cls, in_layers, out_layers, any_layers, info ):
        md = cls( in_layers, out_layers, any_layers )
        md._set_epoch( info['epoch'] )
        md._set_iter( info['iter'] )
        return md
    
    # ------------------ Public attributes ------------------
    
    @property
    def params_(self):
        return self._get_all_params(self._effective_layers_, self._trainable_table_)
        
    @property
    def reg_value_(self):
        return self._get_all_layer_reg_value(self._effective_layers_, self._trainable_table_)
        
    @property
    def inner_updates_(self):
        return self._get_all_inner_updates(self._effective_layers_, self._trainable_table_)
        
    @property
    def in_layers_(self):
        return self._in_layers_
        
    @property
    def out_layers_(self):
        return self._out_layers_
        
    @property
    def any_layers_(self):
        return self._any_layers_
        
    @property
    def in_nodes_(self):
        in_nodes = []
        for layer in self._in_layers_:
            in_nodes += layer.inputs_
        return in_nodes
        
    @property
    def out_nodes_(self):
        return [layer.output_ for layer in self.out_layers_]
        
    @property
    def any_nodes_(self):
        return [layer.output_ for layer in self.any_layers_]
        
    @property
    def gt_nodes_(self):
        return self._gt_nodes_
        
    @property
    def tr_phase_node_(self):
        return self._tr_phase_node_
        
    @property
    def effective_layers_(self):
        return self._effective_layers_
        
    @property
    def trainable_table_(self):
        return self._trainable_table_
        
    @property
    def epoch_(self):
        return self._epoch_
        
    @property
    def iter_(self):
        return self._iter_
        
    @property
    def tr_time_(self):
        return self._tr_time_
        
        
        
class Sequential(Model):
    def __init__(self):
        self._layer_seq_ = []
        
    def add(self, layer):
        self._layer_seq_.append(layer)
        if len(self._layer_seq_) > 1:
            self._layer_seq_[-1].__call__(self._layer_seq_[-2])

    def compile(self):
        md = Model([self._layer_seq_[0]], [self._layer_seq_[-1]])
        md.compile()
        return md
    