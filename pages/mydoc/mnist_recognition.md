---
title: Mnist Recognition
last_updated: Sep 21, 2016
sidebar: mydoc_sidebar
permalink: mnist_recognition.html
toc: false
---

Mnist is a 10 digit dataset with 50000 training samples, 10000 validation samples and 10000 test samples. Each sample is 32*32 pixel. We will show how to build a fully connected neural netork to do digit recognition. The source code can be found here <a href="https://github.com/qiuqiangkong/Hat/blob/master/examples/mnist/main_dnn.py">main_dnn.py</a> and ran without modification. To run it you need <a href="https://github.com/qiuqiangkong/Hat/blob/master/examples/mnist/prepare_data.py">prepare_data.py</a>

<br>
<b>I. Prepare Data</b><br>
Download and load the training & validation & testing data

```
from prepare_data import load_data

### load & prepare data
tr_X, tr_y, va_X, va_y, te_X, te_y = load_data()

# init params
n_in = 784
n_hid = 500
n_out = 10
```

After loading, tr_X, tr_y should be size of (50000,784), (50000,) respectively. 50000 is the number of training sample. 784 is the flattened 32*32 pixel. tr_y is a vector such as [0,4,9,7,...], where the numbers correspond to which class they belong to. We call this form of tr_y as sparse representation. va_X, va_y, te_X and te_y is similar to tr_X and tr_y. 

<br>
<b>II. Categorical Represenation</b><br>
We need to convert the sparse representation of tr_y, va_y, te_y to categorical represenation. 
For example, the vector [0,4,9,7,...] will be converted to <br>
[[1,0,0,0,0,0,0,0,0,0] <br>
 [0,0,0,1,0,0,0,0,0,0] <br>
 [0,0,0,0,0,0,0,0,0,1] <br>
 [0,0,0,0,0,0,0,1,0,0] <br>
...]

```
from hat.preprocessing import sparse_to_categorical

# sparse label to 1-of-K categorical label
tr_y = sparse_to_categorical(tr_y, n_out)
va_y = sparse_to_categorical(va_y, n_out)
te_y = sparse_to_categorical(te_y, n_out)
```

Categorical representation is convenient becuase for a classification problem, each class corresponds to its own output. 

<br>
<b>III. Build the model.</b><br>
```
from hat.models import Sequential
from hat.layers.core import InputLayer, Dense, Dropout

### Build model
seq = Sequential()
seq.add( InputLayer( n_in ) )
seq.add( Dense( n_hid, act='relu' ) )
seq.add( Dropout( p_drop=0.2 ) )
seq.add( Dense( n_hid, act='relu' ) )
seq.add( Dropout( p_drop=0.2 ) )
seq.add( Dense( n_out, act='softmax' ) )
md = seq.combine()

# print summary info of model
md.summary()
```

Sequential means a neural network without bifurcation (Bifurcation neural networks will be introduced later). InputLayer should be the first layer of all neural networks which specify the input attributes. Dense is a fully connected layer. You can specify number of hidden units, activations and other arguments. Dropout layer can drop units according to specified probability. The last layer is a fully connected Dense layer with softmax output. seq.combine() is used to obtain the model. 

<br>
<b>IV. Optimizer</b><br>
Users can choose their favorite optimizer, such as SGD, Adagrad, Rmsprop, Adam, etc. 

```
from hat.optimizers import SGD, Adam

### optimization method
optimizer = Adam( lr=0.001 )        # Try SGD, Adagrad, Rmsprop, etc. instead
```

<br>
<b>V. Callbacks <br></b>
Callbacks are optional but very useful. They are executed every (or several) training epochs are done. Callbacks can be save model, print evaluation result or user defined any functions. 

```
from hat.callbacks import SaveModel, Validation

### callbacks (optional)
# save model every n epoch (optional)
if not os.path.exists('Md'): os.makedirs('Md')
save_model = SaveModel( dump_fd='Md', call_freq=2 )

# validate model every n epoch (optional)
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=te_X, te_y=te_y, batch_size=500, 
                         metrics=['categorical_error'], call_freq=1, dump_path='validation.p' )

# callbacks function
callbacks = [validation, save_model]
```

<br>
<b>VI. Fit model</b><br>
This is part trains the model using provided data. 

```
### train model
md.fit( x=tr_X, y=tr_y, batch_size=500, n_epochs=101, loss_func='categorical_crossentropy', optimizer=optimizer, callbacks=callbacks )
```
Users can tune the parameters. loss_func can be 'mse', 'categorical_crossentropy', 'binary_crossentropy' or user defined loss function. 

<br>
<b>VII. Predict</b><br>
After the model is trained, it can be used to classify new data. 

```
### predict using model
pred_y = md.predict( te_X )
```

<br>
The source code can be found here <a href="https://github.com/qiuqiangkong/Hat/blob/master/examples/mnist/main_dnn.py">main_dnn.py</a> and ran without modification. To run it you need <a href="https://github.com/qiuqiangkong/Hat/blob/master/examples/mnist/prepare_data.py">prepare_data.py</a>

<br>
For more examples, please click the next section <a href="/more_examples.html">More Examples</a>.