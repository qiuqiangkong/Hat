'''
SUMMARY:  prepare data
AUTHOR:   Qiuqiang Kong
Created:  2016.07.22
Modified: 2016.07.27 add download data for publish version
--------------------------------------
'''
import numpy as np
import cPickle
import os
import tarfile

def _load_file( path ):
    data_lb = cPickle.load( open( path, 'rb' ) )
    return data_lb['data'], data_lb['labels']

# load train & test data
def load_data():
    dataset = 'cifar-10-python.tar.gz'
    
    # download data
    if not os.path.isfile(dataset):
        from six.moves import urllib
        print 'downloading data ... (163 Mb)'
        print 'you can also download the data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        urllib.request.urlretrieve( 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', dataset )
    
    # unzip data
    tar = tarfile.open( dataset )
    tar.extractall()
    tar.close()
    print 'extracted successfully!'
    
    # load train data
    data_list, lb_list = [], []
    tr_names = [ 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5' ]
    for na in tr_names:
        data, lbs = _load_file( 'cifar-10-batches-py/' + na )
        data_list.append( data )
        lb_list += lbs
    
    tr_X = np.concatenate( data_list, axis=0 )
    tr_y = np.array( lb_list )
    
    # load test data
    te_X, te_y = _load_file( 'cifar-10-batches-py/test_batch' )
    te_y = np.array( te_y )
    
    return tr_X, tr_y, te_X, te_y