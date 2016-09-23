---
title: Installation
last_updated: Sep 21, 2016
sidebar: mydoc_sidebar
permalink: installation.html
toc: false
---



## Installation


<b>Pre-requsites:</b>
<ul>
  <li>numpy, scipy</li>
  <li>theano</li>
  <li>matplotlib, networkx (plotting neural networks)</li>
</ul>

<br>
<b>Install Hat using pip (recommanded)</b>

```liquid
$ pip install hat
```

For updating the package, you can specify upgrade argument

```liquid
$ pip install hat --upgrade
```

<br>
<b>Using Source Code</b><br>
You can use the source code without installation <a href="https://github.com/qiuqiangkong/Hat">https://github.com/qiuqiangkong/Hat</a>. You need to add following code to append path in every file you use Hat. 

```
import sys
sys.path.append('xxx/Hat')
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

