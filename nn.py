"""
Test your network here!

No need to change this code, but feel free to tweak it
to test your network!

Make your changes to backward method of the Sigmoid class in miniflow.py
"""

import numpy as np
from pyflow import *

from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample

# Load data
data = load_boston()
X_ = data['data']
y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 3
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = Input(value=X_), Output(value=y_)
W1, b1 = Input(value=W1_), Input(value=b1_)
W2, b2 = Input(value=W2_), Input(value=b2_)

l1 = Linear(X, W1, b1) 
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(y, l2)


# feed_dict = {
#     X: X_,
#     y: y_,
#     W1: W1_,
#     b1: b1_,
#     W2: W2_,
#     b2: b2_
# }

epochs = 1000
# Total number of examples
m = X_.shape[0]
batch_size = 11
steps_per_epoch = m // batch_size

# graph = topological_sort(feed_dict)
# trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

# # Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
#         # Step 1
#         # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

# #         # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        # forward_and_backward(graph)
        cost.forward_and_backward()

        # Step 3
        # sgd_update(trainables)
        cost.sgd_update()

        # loss += graph[-1].value
        loss += cost.L[-1].value
    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))