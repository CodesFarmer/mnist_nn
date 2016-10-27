import read_mnist
import network_mnist

if __name__ == "__main__":
    print "Loading data......"
    tr_d, va_d, te_d = read_mnist.load_data_wrapper()

    print "Initizaling the network......"
    nnclf = network_mnist.Network([784, 15, 10, 10])
    print "Training the network......"
    nnclf.SGD(tr_d,100, 10,0.1,te_d)
    #nnclf.RGD(tr_d, 100, 0.01, te_d)
    print "Test the network......"
    n_cor = nnclf.evaluate(te_d)
    print "There are total ", n_cor, " of ", len(te_d), "samples are classified correctly!"