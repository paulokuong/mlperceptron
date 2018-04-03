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
