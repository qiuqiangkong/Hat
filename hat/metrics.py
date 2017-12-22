'''
SUMMARY:  metrics for evaluating
AUTHOR:   Qiuqiang Kong
Created:  2016.05.13
Modified: 2016.05.21 Add prec_recall_fvalue
--------------------------------------
'''
import backend as K
import numpy as np
from sklearn import metrics

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
    err = np.sum(np.not_equal(sp_y_pred, sp_y_gt)) / float(np.prod(sp_y_gt.shape))
    return err
    
def binary_error(p_y_pred, y_gt, thres):
    assert p_y_pred.shape==y_gt.shape, "shape of y_out " + str(p_y_pred.shape) + " and y_gt " + str(y_gt.shape) + " (ground truth) are not equal!"
    (tp, fn, fp, tn) = tp_fn_fp_tn(p_y_pred, y_gt, thres)
    return float(fp+fn) / (tp+tn+fp+fn)
    
'''
categorical crossentropy
size(p_y_pred): N*n_out
'''
def categorical_crossentropy(p_y_pred, y_gt):
    assert len(p_y_pred)==len(y_gt), "Length of y_out and y_gt (ground true) is not equal!"
    N = len(p_y_pred)
    p_y_pred = np.clip(p_y_pred, _EPSILON, 1.-_EPSILON)
    crossentropy = - np.mean(np.sum(y_gt * np.log(p_y_pred), axis=-1))
    return crossentropy
    
def binary_crossentropy(p_y_pred, y_gt):
    assert len(p_y_pred)==len(y_gt), "Length of y_out and y_gt (ground true) is not equal!"
    p_y_pred = np.clip(p_y_pred, _EPSILON, 1.-_EPSILON)
    crossentropy = -np.mean(y_gt * np.log(p_y_pred) + (1.-y_gt) * np.log(1.-p_y_pred))
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

def tp_fn_fp_tn(p_y_pred, y_gt, thres, average):
    """
    Args:
      p_y_pred: shape = (n_samples,) or (n_samples, n_classes)
      y_gt: shape = (n_samples,) or (n_samples, n_classes)
      thres: float between 0 and 1. 
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      tp, fn, fp, tn or list of tp, fn, fp, tn. 
    """
    if p_y_pred.ndim == 1:
        y_pred = np.zeros_like(p_y_pred)
        y_pred[np.where(p_y_pred > thres)] = 1.
        tp = np.sum(y_pred + y_gt > 1.5)
        fn = np.sum(y_gt - y_pred > 0.5)
        fp = np.sum(y_pred - y_gt > 0.5)
        tn = np.sum(y_pred + y_gt < 0.5)
        return tp, fn, fp, tn
    elif p_y_pred.ndim == 2:
        tps, fns, fps, tns = [], [], [], []
        n_classes = p_y_pred.shape[1]
        for j1 in xrange(n_classes):
            (tp, fn, fp, tn) = tp_fn_fp_tn(p_y_pred[:, j1], y_gt[:, j1], thres, None)
            tps.append(tp)
            fns.append(fn)
            fps.append(fp)
            tns.append(tn)
        if average is None:
            return tps, fns, fps, tns
        elif average == 'micro' or average == 'macro':
            return np.sum(tps), np.sum(fns), np.sum(fps), np.sum(tns)
        else: 
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")
        
def prec_recall_fvalue(p_y_pred, y_gt, thres, average):
    """
    Args:
      p_y_pred: shape = (n_samples,) or (n_samples, n_classes)
      y_gt: shape = (n_samples,) or (n_samples, n_classes)
      thres: float between 0 and 1. 
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      prec, recall, fvalue | list or prec, recall, fvalue. 
    """
    eps = 1e-10
    if p_y_pred.ndim == 1:
        (tp, fn, fp, tn) = tp_fn_fp_tn(p_y_pred, y_gt, thres, average=None)
        prec = tp / max(float(tp + fp), eps)
        recall = tp / max(float(tp + fn), eps)
        fvalue = 2 * (prec * recall) / max(float(prec + recall), eps)
        return prec, recall, fvalue
    elif p_y_pred.ndim == 2:
        n_classes = p_y_pred.shape[1]
        if average is None or average == 'macro':
            precs, recalls, fvalues = [], [], []
            for j1 in xrange(n_classes):
                (prec, recall, fvalue) = prec_recall_fvalue(p_y_pred[:, j1], y_gt[:, j1], thres, average=None)
                precs.append(prec)
                recalls.append(recall)
                fvalues.append(fvalue)
            if average is None:
                return precs, recalls, fvalues
            elif average == 'macro':
                return np.mean(precs), np.mean(recalls), np.mean(fvalues)
        elif average == 'micro':
            (prec, recall, fvalue) = prec_recall_fvalue(p_y_pred.flatten(), y_gt.flatten(), thres, average=None)
            return prec, recall, fvalue
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")
        
def prec_recall_fvalue_from_tp_fn_fp(tp, fn, fp, average):
    """
    Args:
      tp, fn, fp: int | list or ndarray of int
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      prec, recall, fvalue
    """
    eps = 1e-10
    if type(tp) == int or type(tp) == np.int32 or type(tp) == np.int64:
        prec = tp / max(float(tp + fp), eps)
        recall = tp / max(float(tp + fn), eps)
        fvalue = 2 * (prec * recall) / max(float(prec + recall), eps)
        return prec, recall, fvalue
    elif type(tp) == list or type(tp) == np.ndarray:
        n_classes = len(tp)
        if average is None or average == 'macro':
            precs, recalls, fvalues = [], [], []
            for j1 in xrange(n_classes):
                (prec, recall, fvalue) = prec_recall_fvalue_from_tp_fn_fp(tp[j1], fn[j1], fp[j1], average=None)
                precs.append(prec)
                recalls.append(recall)
                fvalues.append(fvalue)
            if average is None:
                return precs, recalls, fvalues
            elif average == 'macro':
                return np.mean(precs), np.mean(recalls), np.mean(fvalues)
        elif average == 'micro':
            (prec, recall, fvalue) = prec_recall_fvalue_from_tp_fn_fp(np.sum(tp), np.sum(fn), np.sum(fp), average=None)
            return prec, recall, fvalue
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect type!")
        
def tpr_fpr(p_y_pred, y_gt, thres, average):
    """
    Args:
      p_y_pred: shape = (n_samples,) or (n_samples, n_classes)
      y_gt: shape = (n_samples,) or (n_samples, n_classes)
      thres: float between 0 and 1. 
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      tpr, fpr or list of tpr, fpr. 
    """
    eps = 1e-10
    if p_y_pred.ndim == 1:
        (tp, fn, fp, tn) = tp_fn_fp_tn(p_y_pred, y_gt, thres, average=None)
        tpr = tp / max(float(tp + fn), eps)
        fpr = fp / max(float(fp + tn), eps)
        return tpr, fpr
    elif p_y_pred.ndim == 2:
        n_classes = p_y_pred.shape[1]
        if average is None or average == 'macro':
            tprs, fprs = [], []
            for j1 in xrange(n_classes):
                (tpr, fpr) = tpr_fpr(p_y_pred[:, j1], y_gt[:, j1], thres, average=None)
                tprs.append(tpr)
                fprs.append(fpr)
            if average is None:
                return tprs, fprs
            elif average == 'macro':
                return np.mean(tprs), np.mean(fprs)
        elif average == 'micro':
            (tpr, fpr) = tpr_fpr(p_y_pred.flatten(), y_gt.flatten(), thres, average=None)
            return tpr, fpr
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")
    
def error_rate(pred, gt, thres, average):
    """Error rate. 
    Ref: Mesaros, Annamaria, Toni Heittola, and Tuomas Virtanen. "Metrics for 
       polyphonic sound event detection." Applied Sciences 6.6 (2016): 162.
       
    Args:
      pred: shape = (n_samples,) or (n_samples, n_classes)
      gt: shape = (n_samples,) or (n_samples, n_classes)
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      er: float | list of float. 
    """
    if pred.ndim == 1:
        (tp, fn, fp, tn) = tp_fn_fp_tn(pred, gt, thres, average=None)
        n_substitue = min(fn, fp)
        n_delete = max(0, fn - fp)
        n_insert = max(0, fp - fn)
        n_gt = np.sum(gt)
        if n_gt == 0:
            er = 0.
        else:
            er = (n_substitue + n_delete + n_insert) / float(n_gt)
        return er, n_substitue / float(n_gt), n_delete / float(n_gt), n_insert / float(n_gt)
    elif pred.ndim == 2:
        n_classes = pred.shape[1]
        if average is None or average == 'macro':
            ers, n_subs, n_dels, n_inss = [], [], [], []
            for j1 in xrange(n_classes):
                (er, n_substitue, n_delete, n_insert) = error_rate(pred[:, j1], gt[:, j1], thres, average=None)
                ers.append(er)
                n_subs.append(n_substitue)
                n_dels.append(n_delete)
                n_inss.append(n_insert)
            if average is None:
                return ers, n_subs, n_dels, n_inss
            elif average == 'macro':
                return np.mean(ers), np.mean(n_subs), np.mean(n_dels), np.mean(n_inss)
        elif average == 'micro':
            (er, n_substitue, n_delete, n_insert) = error_rate(pred.flatten(), gt.flatten(), thres, average=None)
            return er, n_substitue, n_delete, n_insert
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")

def error_rate_from_tp_fn_fp(tp, fn, fp, average):
    """
    Args:
      tp, fn, fp: int | list or ndarray of int
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
      
    Returns:
      er: float | list of float. 
    """
    if type(tp) == int or type(tp) == np.int32 or type(tp) == np.int64:
        n_substitue = min(fn, fp)
        n_delete = max(0, fn - fp)
        n_insert = max(0, fp - fn)
        n_gt = tp + fn
        if n_gt == 0:
            er = 0.
        else:
            er = (n_substitue + n_delete + n_insert) / float(n_gt)
        return er, n_substitue / float(n_gt), n_delete / float(n_gt), n_insert / float(n_gt)
    elif type(tp) == list or type(tp) == np.ndarray:
        n_classes = len(tp)
        if average is None or average == 'macro':
            ers, n_subs, n_dels, n_inss = [], [], [], []
            for j1 in xrange(n_classes):
                (er, n_substitue, n_delete, n_insert) = error_rate_from_tp_fn_fp(tp[j1], fn[j1], fp[j1], average=None)
                ers.append(er)
                n_subs.append(n_substitue)
                n_dels.append(n_delete)
                n_inss.append(n_insert)
            if average is None:
                return ers, n_subs, n_dels, n_inss
            elif average == 'macro':
                return np.mean(ers), np.mean(n_subs), np.mean(n_dels), np.mean(n_inss)
        elif average == 'micro':
            (er, n_substitue, n_delete, n_insert) = error_rate_from_tp_fn_fp(np.sum(tp), np.sum(fn), np.sum(fp), average=None)
            return er, n_substitue, n_delete, n_insert
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")

def equal_error_rate(p_y_pred, y_gt, average):
    """Equal error rate. 
    Modified from Peter Foster's eer.py
    
    Args:
      p_y_pred: shape = (n_samples,) or (n_samples, n_classes)
      y_gt: shape = (n_samples,) or (n_samples, n_classes)
      average: None (element wise) | 'micro' (calculate metrics globally) 
        | 'macro' (calculate metrics for each label then average). 
        
    Returns:
      eer | list of eer. 
    """
    if p_y_pred.ndim == 1:
        fpr, tpr, thresholds = metrics.roc_curve(y_gt, p_y_pred, drop_intermediate=True)
        
        eps = 1E-6
        Points = [(0,0)]+zip(fpr, tpr)
        for i, point in enumerate(Points):
            if point[0]+eps >= 1-point[1]:
                break
        P1 = Points[i-1]; P2 = Points[i]
            
        #Interpolate between P1 and P2
        if abs(P2[0]-P1[0]) < eps:
            eer = P1[0]        
        else:        
            m = (P2[1]-P1[1]) / (P2[0]-P1[0])
            o = P1[1] - m * P1[0]
            eer = (1-o) / (1+m)  
        return eer
    elif p_y_pred.ndim == 2:
        n_classes = p_y_pred.shape[1]
        if average is None or average == 'macro':
            eers = []
            for j1 in xrange(n_classes):
                eer = equal_error_rate(p_y_pred[:, j1], y_gt[:, j1], average=None)
                eers.append(eer)
            if average is None:
                return eers
            elif average == 'macro':
                return np.mean(eer)
        elif average == 'micro':
            eer = equal_error_rate(p_y_pred.flatten(), y_gt.flatten(), average=None)
            return eer
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")

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