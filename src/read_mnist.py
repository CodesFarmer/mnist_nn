import cPickle
import gzip

import numpy as np

def load_data():

    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():

    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_labels = [vectorized_result(y) for y in tr_d[1]]
    training = zip(training_inputs, training_labels)

    validation_inputs = [np.reshape(x, (784,1)) for x in va_d[0]]
    validation_labels = va_d[1]
    validation = zip(validation_inputs, validation_labels)

    test_inputs = [np.reshape(x, (784,1)) for x in te_d[0]]
    test_labels = te_d[1]
    test = zip(test_inputs, test_labels)

    return (training, validation, test)

def vectorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e