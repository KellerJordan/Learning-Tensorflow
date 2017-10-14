import tensorflow as tf
import numpy as np

theta = np.random.uniform(0, 2)
thetap = np.random.uniform(0, 2)
print(theta, thetap)
X = np.random.randn(10000, 1)
Y = np.zeros(X.shape)
for i, x in enumerate(X):
    lmbda = np.exp(thetap * x)
    Y[i] = np.random.poisson(lmbda)

print(np.mean(Y))

