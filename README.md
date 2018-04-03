[![Build Status](https://travis-ci.org/paulokuong/neural_network.svg?branch=master)](https://travis-ci.org/paulokuong/neural_network)[![Coverage Status](https://coveralls.io/repos/github/paulokuong/neural_network/badge.svg?branch=master)](https://coveralls.io/github/paulokuong/neural_network?branch=master)
Multilayer Neural Network in Python
==================

Python implementation of multilayer perceptron neural network from scratch.

> Minimal neural network class with regularization using scipy minimize. Contains clear pydoc for learners to better understand each stage in the neural network.

Requirements
------------

* Python 3.4 (tested)

Installation
------------
```
    pip install neuralnetwork
```

Goal
----

To provide an example of a simple MLP for educational purpose.

Code sample
-----------

Predicting outcome of AND logic gate:

X = 000, 001, 010, 011, 100, 101, 110, 111
y = 0,0,0,0,0,0,1

Data we want to predict:
p = 011, 111, 000, 010, 111
Expected results are: 0, 1, 0, 0, 1

```python

    import numpy as np
    from neural_network import NeuralNetwork

    X = np.matrix(
        '0 0 0;0 0 1;0 1 0;0 1 1;1 0 0;1 0 1;1 1 0;1 1 1')
    y = np.matrix('0;0;0;0;0;0;0;1')
    n = NeuralNetwork((5,5,))

    g = n.train(X, y, 0.01, show_cost=True)
    y_pred = n.predict(np.matrix('0 1 1;1 1 1;0 0 0;0 1 0;1 1 1'), g)

    print(y_pred)
    print(n.accuracy(y_pred, np.matrix('0;1;0;0;1')))
```

Result
```
Loss: 1.150514159559136
Loss: 1.1505141504773504
Loss: 1.1505141583532519
Loss: 1.1505141469667088
Loss: 1.2434056388432244
Loss: 0.8288594235909958
Loss: 0.8288594198770226
Loss: 0.8288594228018156
Loss: 0.7807394496223645
....
....
Loss: 0.11768930150540982
Loss: 0.11768930150555365
Loss: 0.11768930150553429
Loss: 0.1176893015053323
Loss: 0.11768328608538177
Loss: 0.1176832860852971
Loss: 0.11768328608527757
Loss: 0.11768328608535958
Loss: 0.1176832860853596
Loss: 0.11768328608534921
Loss: 0.1176832860853433
Loss: 0.11768197870406577
Loss: 0.11768197870406577
[[0]
 [1]
 [0]
 [0]
 [1]]
1.0
```


Contributors
------------

* Paulo Kuong ([@pkuong](https://github.com/paulokuong))
