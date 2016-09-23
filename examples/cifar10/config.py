'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.07.22
Modified: -
--------------------------------------
'''
data_fd = '/user/HS229/qk00006/Datasets/cifar10/cifar-10-batches-py'
tr_names = [ 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5' ]

lbs = [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
lb_to_ix = { lb:i for i, lb in enumerate(lbs) }
ix_to_lb = { i:lb for i, lb in enumerate(lbs) }
