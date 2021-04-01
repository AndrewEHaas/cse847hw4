import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, TypeVar

Array = TypeVar('Array')


# ==== Additional Methods ==== #

sigmoid = lambda data : 1/(1 + np.exp(-1*data))
predict = lambda data, weights : np.where(sigmoid(data @ weights) > .5, 1, 0)

def loss(data: Array, weights: Array, labels: Array) -> float:
    s = sigmoid(data @ weights)
    return (-1/data.shape[0])*np.sum(labels*np.log(s) + (1-labels)*np.log(1-s))

def gradient(data: Array, weights: Array, labels: Array) -> Array:
    s = sigmoid(data @ weights)
    g = np.zeros(weights.shape)
    for i in range(data.shape[0]):
        g += np.reshape((labels[i,:] - s[i,:])*data[i,:], g.shape)
    return (-1/data.shape[0])*g

def l1_loss(data: Array, weights: Array, labels: Array, par: float) -> float:
    return loss(data, weights, labels) + par*np.linalg.norm(weights, ord=1)/data.shape[0]

def l1_gradient(data: Array, weights: Array, labels: Array, par: float) -> float:
    return gradient(data, weights, labels) + par*np.sign(weights)/data.shape[0]

# ======================== #

def logistic_train(data: Array, labels: Array, lr: float, epsilon: float = 1e-5, maxiter: int = 1000) -> Array:
    """
    parameters:
        data   : n * (d + 1) with n samples and d features, where col d+1 is all ones (corresponding to intercept)

        labels : n * 1 vector of class labels (0 or 1)

        lr : learning rate

        epsilon: optional parameter specifying convergence - if the change in absolute difference in predictions, from one iteration to the next
                 when averaged across input features is less than epsilon, halt (default 1e-5)

        maxiter: optional parameter that specifies the maximum number of iterations (default 1000)

    return: 1 * (d + 1) vector of weights corresponding to columns of data - last entry is bias term
    """
    # instantiate weight vector, store previous weight iteration
    weights = np.zeros((data.shape[1],1))
    prev = np.copy(weights)

    #ls = [] # list to store loss values at each iteration

    for iter in range(maxiter):

        # update weights
        weights -= lr*gradient(data, weights, labels)

        # compute and store loss
        #ls.append(loss(data, weights, labels))

        # check break condition, halting of necessary
        if np.linalg.norm(prev - weights, ord=2) < epsilon:
            return weights
        prev = np.copy(weights)

    #plt.plot(range(maxiter), ls, label='learning_rate: ' + str(lr))
    #plt.xlabel('iteration')
    #plt.ylabel('loss function')
    #plt.legend(loc='best')
    #plt.savefig('test.jpg')
    return weights

def logistic_l1_train(data: Array, labels: Array, par: float, lr:float, epsilon: float = 1e-5, maxiter: int = 5000):
    """
    parameters:
        data   : n * (d + 1) with n samples and d features, where col d+1 is all ones (corresponding to intercept)

        labels : n * 1 vector of class labels (0 or 1)

        par : regularization parameter

        lr : learning rate

        epsilon: optional parameter specifying convergence - if the change in absolute difference in predictions, from one iteration to the next
                 when averaged across input features is less than epsilon, halt (default 1e-5)

        maxiter: optional parameter that specifies the maximum number of iterations (default 1000)

    return: 1 * (d + 1) vector of weights corresponding to columns of data - last entry is bias term
    """
    # instantiate weight vector, store previous weight iteration
    weights = np.zeros((data.shape[1],1))
    prev = np.copy(weights)

    #ls = [] # list to store loss values

    for iter in range(maxiter):

        # update weights
        weights -= lr*l1_gradient(data, weights, labels, par)

        # compute and store loss
        #ls.append(l1_loss(data, weights, labels, par))

        # check break condition, halting if necessary
        if np.linalg.norm(prev - weights, ord=2) < epsilon:
            return weights
        prev = np.copy(weights)

    #plt.plot(range(maxiter), ls, label='par: ' + str(par))
    #plt.xlabel('iteration')
    #plt.ylabel('loss function')
    #plt.legend(loc='best')
    #plt.savefig('test.jpg')
    return weights
