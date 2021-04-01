import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import prettytable as pt
from HW4 import sigmoid, predict, loss, gradient, logistic_train, logistic_l1_train

def test1a():
    # simply looks at convergence for various parameters of dataset 1

    # read data
    data = np.loadtxt('data_spam.txt')
    labels = np.loadtxt('labels_spam.txt')

    # append column of 1 ones for bias term
    data = np.concatenate((data, np.ones((data.shape[0],1))), axis=1)

    # reshape labels
    labels = np.reshape(labels, (4601, 1))

    plt.figure(figsize=(16,13), dpi=100)
    for lr in [.1, .01, .001, .0001]:
        logistic_train(data, labels, lr)

    #[0, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    plt.figure(figsize=(16,13), dpi=100)
    for par in [0, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
        logistic_l1_train(data, labels, par, .01, maxiter=5000)

def test1b():
    # tests accuracy with various training sets

    # read data
    data = np.loadtxt('data_spam.txt')
    labels = np.loadtxt('labels_spam.txt')

    # append column of 1 ones for bias term
    data = np.concatenate((data, np.ones((data.shape[0],1))), axis=1)

    # reshape labels
    labels = np.reshape(labels, (4601, 1))

    # split data and labels for testing
    x_test, y_test = data[2000:, :], labels[2000:, :]

    table = pt.PrettyTable()
    table.field_names = ['Training Size', 'Testing Misclassifications']

    for training_size in [200, 500, 800, 1000, 1500, 2000]:
        w = logistic_train(data[:training_size,:], labels[:training_size], .05, maxiter=7500)
        prediction = predict(x_test, w)
        temp = 0
        for i in range(y_test.shape[0]):
            if prediction[i,:] != y_test[i,:]:
                temp += 1
        table.add_row([training_size, temp])

    print(table)


def test2():
    # read data
    data = loadmat('ad_data.mat')
    x_train, y_train = data['X_train'], data['y_train']
    x_test, y_test = data['X_test'], data['y_test']

    # change encoding
    y_train = np.where(y_train == 1, 1, 0)
    y_test  = np.where(y_test == 1, 1, 0)

    table = pt.PrettyTable()
    table.field_names = ['Reg Param', 'Testing Misclassifications', 'Nonzero Weights']
    for par in [0, .01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
        w = logistic_l1_train(x_train, y_train, par, .01, epsilon=1e-20, maxiter=5000)
        prediction = predict(x_test, w)

        # count misclassifications
        temp1 = 0
        for i in range(y_test.shape[0]):
            if prediction[i,:] != y_test[i,:]:
                temp1 += 1
        # count nonzero weights
        temp2 = 0
        for i in range(w.shape[0]):
            if abs(w[i,:]) > 1e-3:
                temp2 += 1

        table.add_row([par, temp1, temp2])

    print(table)

test2()
