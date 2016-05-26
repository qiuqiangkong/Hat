import numpy as np
import pickle
import matplotlib.pyplot as plt

def plot_loss_err( path='validation.p', print_type='err' ):
    Result = pickle.load( open( path, 'rb' ) )
    if print_type=='err':
        tr = Result['tr_errs']
        va = Result['va_errs']
        te = Result['te_errs']
    if print_type=='loss':
        tr = Result['tr_losses']
        va = Result['va_losses']
        te = Result['te_losses']
        
    valid_freq = Result['call_freq']
    
    line_tr, = plt.plot(tr, label='tr_'+print_type, color='b')
    line_va, = plt.plot(va, label='va_'+print_type, color='g')
    line_te, = plt.plot(te, label='te_'+print_type, color='r')
    
    N = len(tr)
    plt.xticks( np.arange(N), np.arange(0,N*valid_freq, valid_freq) )
    plt.legend(handles=[line_tr, line_va, line_te])
    if print_type=='err': plt.axis([0, N , 0, 1])
    plt.show()