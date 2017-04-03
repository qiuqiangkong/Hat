import backend as K
import numpy as np


def loss_func(out_nodes, inter_nodes, gt_nodes):
    """User can define their own objective function
    """
    # return loss_node
    pass


### objectives, all return are graphic representation
# _EPSILON = 1e-8     # if set smaller, some objective function will underflow
_EPSILON = 1e-7

# categorical crossentropy
def categorical_crossentropy(p_y_pred, y_gt):
    p_y_pred = K.clip(p_y_pred, _EPSILON, 1. - _EPSILON)
    return K.mean(K.categorical_crossentropy(p_y_pred, y_gt))
    
def sparse_categorical_crossentropy(p_y_pred, y_gt):
    p_y_pred = K.clip(p_y_pred, _EPSILON, 1. - _EPSILON)
    y_gt = K.to_one_hot(y_gt, )
    return K.mean(K.categorical_crossentropy(p_y_pred, y_gt))
    
# binary crossentropy
def binary_crossentropy(p_y_pred, y_gt):
    p_y_pred = K.clip(p_y_pred, _EPSILON, 1. - _EPSILON)    
    return K.mean(K.mean(K.binary_crossentropy(p_y_pred, y_gt), axis=-1))
    
# mean square error, mean(||y-o||_{2}^{2})
def mse(y_pred, y_gt):
    return K.mean(K.sqr(y_pred - y_gt))

# mean lp norm, mean(||y-o||_{p}^{p})
def norm_lp(y_pred, y_gt, norm):
    return K.mean(K.sum(K.power(K.abs(y_pred - y_gt), norm), axis=-1))
    
def norm_l1(y_pred, y_gt):
    return norm_lp(y_pred, y_gt, 1.)
    
# kl divergence, o*log(o/y) - o + y
def kl_divergence(y_pred, y_gt):
    y_pred = K.clip(y_pred, _EPSILON, np.inf)
    y_gt = K.clip(y_gt, _EPSILON, np.inf)
    kl_mat = y_gt * K.log(y_gt / y_pred) - y_gt + y_pred
    return K.mean(K.sum(kl_mat, axis=-1))

# IS divergence, o/y - log(o/y) - 1
def is_divergence(y_pred, y_gt):
    y_pred = K.clip(y_pred, _EPSILON, np.inf)
    y_gt = K.clip(y_gt, _EPSILON, np.inf)
    is_mat = y_gt / y_pred - K.log(y_gt / y_pred) - 1
    return K.mean(K.sum(is_mat, axis=-1))

# beta divergence, beta=R\{0,1}, 1/(beta*(beta-1)) * (o^beta + (beta-1)*y^beta - beta*o*y^(beta-1))
def beta_divergence(y_pred, y_gt, beta):
    y_pred = K.clip(y_pred, _EPSILON, np.inf)
    y_gt = K.clip(y_gt, _EPSILON, np.inf)
    beta_mat = 1. / (beta*(beta-1)) * (K.power(y_gt, beta) + (beta-1) * K.power(y_pred, beta) - beta * y_gt * K.power(y_pred, (beta-1)))
    return K.mean(K.sum(beta_mat, axis=-1))
    
### metrics
def categorical_error(p_y_pred, y_gt):
    y_pred_sparse = K.argmax(p_y_pred, axis=-1)
    y_gt_sparse = K.argmax(y_gt, axis=-1)
    return K.mean(K.neq(y_pred_sparse, y_gt_sparse))
    
def get(loss):
    f = globals().get(loss)
    if f is None:
        
        raise Exception('No ' + loss + ' loss!')
    else:
        return f
        
        
        
