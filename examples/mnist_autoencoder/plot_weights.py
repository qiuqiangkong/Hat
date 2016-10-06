'''
SUMMARY:  plot 1-st autoencoder learned weights
AUTHOR:   Qiuqiang Kong
Created:  2016.10.06
Modified: -
--------------------------------------
'''
from hat import serializations
import matplotlib.pyplot as plt

# load weights of 1-st layer
md = serializations.load( 'Results/md_ae1.p' )
W = md.find_layer( 'a1' ).W_

# plot autoencoder learned weights
num_to_plot = 10
for i1 in xrange( num_to_plot ):
    w = W[:,i1].reshape((28,28))
    plt.matshow(w)
    plt.show()
