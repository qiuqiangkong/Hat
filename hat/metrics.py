'''
SUMMARY:  metrics for evaluating
AUTHOR:   Qiuqiang Kong
Created:  2016.05.13
Modified: 2016.05.21 Add prec_recall_fvalue
--------------------------------------
'''
import backend as K
import numpy as np

_EPSILON = 1e-6     # when set to 1e-8, binary_crossentropy underflow

'''
categorical error
size(p_y_pred): N*n_out
'''
def categorical_error(p_y_pred, y_gt):
    assert len(p_y_pred)==len(y_gt), "Length of y_out and y_gt (ground true) is not equal!"
    N = len(p_y_pred)
    sp_y_pred = np.argmax(p_y_pred, axis=-1)
    sp_y_gt = np.argmax(y_gt, axis=-1)
    # err = np.sum(np.not_equal(sp_y_pred, sp_y_gt)) / float(N)
    err = np.sum(np.not_equal(sp_y_pred, sp_y_gt)) / float(np.prod(sp_y_gt.shape))
    return err
    
def binary_error(p_y_pred, y_gt, thres):
    assert p_y_pred.shape==y_gt.shape, "shape of y_out " + str(p_y_pred.shape) + " and y_gt " + str(y_gt.shape) + " (ground truth) are not equal!"
    tp, tn, fp, fn = tp_tn_fp_fn(p_y_pred, y_gt, thres)
    return float(fp+fn) / (tp+tn+fp+fn)
    
'''
categorical crossentropy
size(p_y_pred): N*n_out
'''
def categorical_crossentropy(p_y_pred, y_gt):
    assert len(p_y_pred)==len(y_gt), "Length of y_out and y_gt (ground true) is not equal!"
    N = len(p_y_pred)
    p_y_pred = np.clip(p_y_pred, _EPSILON, 1.-_EPSILON)
    # crossentropy = -np.sum(y_gt * np.log(p_y_pred)) / float(N)
    crossentropy = np.mean(np.sum(y_gt * np.log(p_y_pred), axis=-1))
    return crossentropy
    
def binary_crossentropy(p_y_pred, y_gt):
    assert len(p_y_pred)==len(y_gt), "Length of y_out and y_gt (ground true) is not equal!"
    N = len(p_y_pred)
    p_y_pred = np.clip(p_y_pred, _EPSILON, 1.-_EPSILON)
    crossentropy = -np.sum(y_gt * np.log(p_y_pred) + (1-y_gt) * np.log(1-p_y_pred)) / float(N)
    return crossentropy
    
def mse(y_pred, y_gt):
    assert len(y_pred)==len(y_gt), "Length of y_out and y_gt (ground true) is not equal!"
    return np.mean(np.sum(np.square(y_pred - y_gt), axis=1))
    
def norm_lp(y_pred, y_gt, norm):
    return np.mean(np.sum(np.power(np.abs(y_pred - y_gt), norm), axis=-1))
    
def norm_l1(y_pred, y_gt):
    return norm_lp(y_pred, y_gt, 1.)

def kl_divergence(y_pred, y_gt):
    y_pred = np.clip(y_pred, _EPSILON, 1. - _EPSILON)
    y_gt = np.clip(y_gt, _EPSILON, 1. - _EPSILON)
    return np.mean(np.sum(y_gt * np.log(y_gt / y_pred) - y_gt + y_pred, axis=-1))

# def tp_tn_fp_fn(p_y_pred, y_gt, thres):
#     y_pred = np.zeros_like(p_y_pred)
#     y_pred[ np.where(p_y_pred>thres) ] = 1.
#     tp = np.sum(y_pred + y_gt > 1.5)
#     tn = np.sum(y_pred + y_gt < 0.5)
#     fp = np.sum(y_pred - y_gt > 0.5)
#     fn = np.sum(y_gt - y_pred > 0.5)
#     return tp, tn, fp, fn

def prec_recall_fvalue(p_y_pred, y_gt, thres):
    tp, tn, fp, fn = tp_tn_fp_fn(p_y_pred, y_gt, thres)
    prec = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    fvalue = 2 * (prec * recall) / (prec + recall)
    return prec, recall, fvalue
    
def tpr_fpr(p_y_pred, y_gt, thres):
    tp, tn, fp, fn = tp_tn_fp_fn(p_y_pred, y_gt, thres)
    tpr = tp / float(tp + fn)
    fpr = fp / float(fp + tn)
    return tpr, fpr
    

def confusion_matrix(p_y_pred, y_gt):
    assert len(p_y_pred)==len(y_gt), "Length of y_out and y_gt (ground true) is not equal!"
    N, n_out = p_y_pred.shape
    sp_y_pred = np.argmax(p_y_pred, axis=-1)
    sp_y_gt = np.argmax(y_gt, axis=-1)
    confM = np.zeros((n_out, n_out))
    for i1 in xrange(N):
        confM[ sp_y_gt[i1], sp_y_pred[i1] ] += 1
    return confM
    
def get(metrics):
    f = globals().get(metrics)
    if f is None:
        raise Exception('No ' + metrics + ' metrics!')
    else:
        return f