import read_mnist
import nnetwork

print "Loading the dataset......"
tr_d, va_d, te_d = read_mnist.load_data_wrapper()

print "Initializing the network......"
nnclf = nnetwork.Network([784, 15, 10])

print "Training the network......"
nnclf.TrainNet(tr_d,va_d,20, 0.1, "SGD")

print "Evaluating on test data......"
print "{0}/{1} on test data !".format(nnclf.Evaluate(te_d), len(te_d))