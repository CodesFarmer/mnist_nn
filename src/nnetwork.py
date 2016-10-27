import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        #The number of layers
        self.sizes = sizes
        self.numlayers = len(sizes)
        #define the network by allocate its weights and biases
        #The number of biases are equal to the number of neurons except the first layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        """
        The number of weights are equal to n(i)*n(i+1), where i is the serial number of layer.
        As we can see from above description, the last layer doesn't have weights
        """
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:], sizes[:-1])]
    def TrainNet(self,tr_d,va_d,epochs,lr,mode):
        """
        :param tr_d: training data
        :param va_d: validation data
        :param epochs: the number of max epochs for training
        :param lr: the learning rate
        :param mode: the approach for training, BGD or SGD
        """
        if(mode == "BGD"):
            self.BGD(tr_d,va_d,epochs,lr)
        elif(mode == "SGD"):
            self.SGD(tr_d,va_d,epochs,lr)
        else:
            print "Wrong mode !"
    def BGD(self, tr_d, va_d, epochs, lr):
        """
        This function optimize the network by all training data
        """
        print "Training the network with BGD......"
        trlen = len(tr_d)
        for j in xrange(epochs):
            random.shuffle(tr_d)
            self.update_network(tr_d, lr)
            #for i in xrange(trlen):
            #   self.update_network([tr_d[i]],lr)
            if(va_d):
                print "Epoch {0}: {1}/{2}".format(j, self.Evaluate(va_d),len(va_d))
            else:
                print "Epoch {0}:".format(j)
    def SGD(self, tr_d, va_d, epochs, lr):
        """
        This function realizes the stochastic gradient descent
        First, we split the dataset into m batches, the we update the
        network based on those batches
        """
        print "Training the network with SGD......"
        trlen = len(tr_d)
        batch_size = 1
        j = 0
        while j < epochs:
            random.shuffle(tr_d)
            batches = [tr_d[k:k+batch_size] for k in xrange(0,trlen,batch_size)]
            for tr_batch in batches:
                self.update_network(tr_batch,lr)
            if (va_d):
                print "Epoch {0}: {1}/{2}".format(j, self.Evaluate(va_d), len(va_d))
            else:
                print "Epoch {0}:".format(j)
            j += 1

    def update_network(self, tr_d, lr):
        """
        Update the network by execute a feed forward step and a back propagation step
        on every training sample
        """
        trlen = len(tr_d)
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in tr_d:
            delta_b_single, delta_w_single = self.backppg(x,y)
            delta_b = [db+dbs for db,dbs in zip(delta_b, delta_b_single)]
            delta_w = [dw+dws for dw,dws in zip(delta_w, delta_w_single)]
        #update the parameters in network
        self.biases = [b - (lr/trlen)*db for b,db in zip(self.biases, delta_b)]
        self.weights = [w - (lr/trlen)*dw for w,dw in zip(self.weights, delta_w)]
    def backppg(self,x,y):
        """
        This function execute a feed forward step, then we can get the residual between
        output and the expected output
        Then a back propagation step are execute to calculate the delta_w and delta_b
        """
        #feed forward
        activation = x
        activations = [x]
        zs = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmod(z)
            activations.append(activation)

        #back propagation, start from the last layer
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        #The residual of last layer equal (a[l]-y)*f'(z[l])
        delta = (activations[-1]-y)*sigmod_deri(zs[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2,self.numlayers):
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sigmod_deri(zs[-l])
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (delta_b,delta_w)
    def Evaluate(self,te_d):
        ncor = [(np.argmax(self.forward(x)), y) for x,y in te_d]
        #for x,y in te_d:
        #    ncor.append([(np.argmax(self.forward(x)), y)])
        #print "Size: ", ncor.shape
        return sum(int(x==y) for x,y in ncor)
    def forward(self,x):
        a = x
        for b,w in zip(self.biases, self.weights):
            a = sigmod(np.dot(w, a) + b)
        return a
def sigmod(z):
    return 1.0/(1.0+np.exp(-z))
def sigmod_deri(z):
    return sigmod(z)*(1-sigmod(z))
def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
def tanh_deri(z):
    return 1-tanh(z)*tanh(z)