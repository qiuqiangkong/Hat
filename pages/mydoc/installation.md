---
title: Installation
last_updated: Sep 21, 2016
sidebar: mydoc_sidebar
permalink: installation.html
toc: false
---



Hat is an open source deep learning framework. We would like to thank Keras, from which we folked a lot of codes. Keras is excellent, with clear code and structure. Compared with Keras, Hat is a deep learning framework provides more functions for (but not limited to) audio processing. Hat allows users to build their neural network in a more flexible way, with a lot of example codes for audio processing. 


## Installation


<b>Pre-requsites:</b>
<ul>
  <li>numpy, scipy</li>
  <li>theano</li>
  <li>matplotlib, networkx (plotting neural networks)</li>
</ul>

<br>
<b>Install Hat using pip</b>

```liquid
$ pip install hat
```

For updating the package, you can specify upgrade argument

```liquid
$ pip install hat --upgrade
```

## Write your first NN
Once you have installed Hat, you can use the following code to build your first NN. 
```
from hat.layers.core import InputLayer, Dense, Dropout
from hat.models import Sequential

seq = Sequential()
seq.add( InputLayer(784) )
seq.add( Dense( n_out=500, act='tanh' ) )
seq.add( Dense( n_out=10, act='softmax' ) )
md = seq.combine()
```

<br>
Congratulations! You have finished your first neural network! But we have not use data to trian it yet. Next section will show how to do write a real digit recognition program. Please click <a href="/mnist_recognition.html">Mnist Recognition</a>.

