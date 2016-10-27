"""
This program expend the mnist data by shift the original image in four different directions
"""

from __future__ import print_function

import cPickle
import gzip
import os.path
import random

import numpy as np

print("We are going to expand the mnist data......")

if os.path.exists("../data/mnist_expanded.pkl.gz"):
    print("The expanded dataset already existed......")
else:
    f = gzip.open("../data/mnist.pkl.gz",'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    expanded_training_pairs = []
    j = 0
    for x,y in zip(training_data[0], training_data[1]):
        expanded_training_pairs.append((x,y))
        image = np.reshape(x, (-1, 28))
        j += 1
        if(j%1000==0):
            print("Expanding image number ",j)

        for d, axis, index_position, index in[
                (1, 0, "first", 0)
                (-1, 0, "first", 27)
                (1, 1, "last", 0)
                (-1, 1, "last", 27)]:
            new_img = np.roll(image, d, axis)
            if index_position == "first":
                new_img[index, :] = np.zeros(28)#Add a black line in first or last row
            else:
                new_img[:, index] = np.zeros(28)#Add a black line in first or last colmun
            expanded_training_pairs.append((np.reshape(new_img, 784), y))
    random.shuffle(expanded_training_pairs)#mixture the expanded images
    expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
    print("Saving the expanded data......")
    f = gzip.open("../data/mnist_expanded.pkl.gz", "w")
    cPickle.dump(expanded_training_data, validation_data, test_data, f)
    f.close()