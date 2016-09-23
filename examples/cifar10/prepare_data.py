'''
SUMMARY:  prepare data
AUTHOR:   Qiuqiang Kong
Created:  2016.07.22
Modified: 2016.09.19 Add scaler
--------------------------------------
'''
import numpy as np
import config as cfg
import cPickle
from PIL import Image
from sklearn import preprocessing

def _load_file( path ):
    data_lb = cPickle.load( open( path, 'rb' ) )
    return data_lb['data'], data_lb['labels']

# load train & test data
def load_data():
    # load train data
    data_list, lb_list = [], []
    for na in cfg.tr_names:
        data, lbs = _load_file( cfg.data_fd + '/' + na )
        data_list.append( data )
        lb_list += lbs
    
    tr_X = np.concatenate( data_list, axis=0 )
    tr_y = np.array( lb_list )
    
    # load test data
    te_X, te_y = _load_file( cfg.data_fd + '/test_batch' )
    te_y = np.array( te_y )
    
    return tr_X, tr_y, te_X, te_y
    
# get scaler of all pixels in all pictures
def get_scaler( x ):
    scaler = preprocessing.StandardScaler().fit( x.astype(np.float32).reshape(-1,1) )
    return scaler
    
# transform according to scaler
def transform( x, scaler ):
    return scaler.transform( x.astype(np.float32) )