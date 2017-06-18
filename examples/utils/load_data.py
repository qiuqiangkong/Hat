"""
SUMMARY:  Datasets
Ref:      http://deeplearning.net/tutorial/code/logistic_sgd.py
          https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
Created:  2016.05.01
Modified: 2017.06.18
--------------------------------------
"""
import numpy as np
import cPickle
import gzip
import os

def load_mnist():
    dataset = 'mnist.pkl.gz'
    if not os.path.isfile(dataset):
        from six.moves import urllib
        print 'downloading data ... (16.2 Mb)'
        urllib.request.urlretrieve( 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', dataset )
        
    f = gzip.open( dataset, 'rb' )
    train_set, valid_set, test_set = cPickle.load(f)
    [tr_X, tr_y] = train_set
    [va_X, va_y] = valid_set
    [te_X, te_y] = test_set
    f.close()
    return tr_X, tr_y, va_X, va_y, te_X, te_y
    

def load_imdb(path="imdb.pkl", nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):
    """
    Load imdb data, from https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
    oov: out-of vocabulary
    """

    if not os.path.isfile(path):
        from six.moves import urllib
        print 'downloading data ... (15.3 Mb)'
        urllib.request.urlretrieve( 'https://s3.amazonaws.com/text-datasets/imdb.pkl', path )

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    X, labels = cPickle.load(f)
    f.close()

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                        'Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X = np.array([np.array(e) for e in X])
    

    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])

    X_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])

    return X_train, y_train, X_test, y_test