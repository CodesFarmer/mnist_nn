"""
This program apply the neural network to classify the images
"""

import random

import numpy as np

def sigmod(x):
    return 1.0/(1.0+np.exp(-x))

def sigmod_prime(x):
    return sigmod(x)*(1-sigmod(x))

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]#A [n1 n2 n3] network has n2+n3 biases
        #A [n1 n2 n3] network has n1*n2+n2*n3 weights
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmod(np.dot(w,a)+b)
        return a

    def SGD(self, tr_d, epochs, mini_batch_size, eta, te_d=None):

        """
        update the network in epochs times
        each time there are feedforward and backward propagation
        :param tr_d:
        :param epochs:
        :param mini_batch_size:
        :param eta:
        :param te_d:
        :return:
        """

        if te_d:n_test = len(te_d)
        n = len(tr_d)
        for j in xrange(epochs):
            random.shuffle(tr_d)
            #mixture the training data and split it into n/mini_batch_size block
            mini_batches = [tr_d[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if te_d:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(te_d), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]#calculate the sum of residual about bias
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]#calculate the sum of residual about weights
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]#update the bias
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]#update the weights

    def backprop(self, x, y):
        #first, we execute the feedforward in network
        #and record the intermediate results
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #forward
        activation = x
        activations = [x]
        zs = []#it is the middle value in network
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b             #z=w*x+b
            zs.append(z)
            activation = sigmod(z)#activation function,a=1/(1+exp(-z)), it is also the inputs of next layer
            activations.append(activation)

        #back propagation
        delta = self.cost_derivative(activations[-1], y)*sigmod_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmod_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):

        return (output_activations-y)

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x==y) for (x, y) in test_results)

    def RGD(self, tr_d, epochs, eta, te_d=None):
        if te_d:
            n_test = len(te_d)
        n = len(tr_d)
        for j in xrange(epochs):
            random.shuffle(tr_d)
            for x,y in tr_d:
                nabla_b, nabla_w = self.backprop(x,y)
                self.biases = [b - eta*nb for b, nb in zip(self.biases, nabla_b)]
                self.weights = [w - eta*nw for w, nw in zip(self.weights, nabla_w)]

            if te_d:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(te_d), n_test)
            else:
                print "Epoch {0} complete".format(j)