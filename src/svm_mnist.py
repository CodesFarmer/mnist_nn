"""
This program simply employing the SVM on dataset without any pre-process
"""
import read_mnist
from sklearn import svm

def svm_baseline():
    tr_d, va_d, te_d = read_mnist.load_data()
    #training stage
    clf = svm.SVC()
    clf.fit(tr_d[0], tr_d[1])
    #test stage
    predictions = [int(a) for a in clf.predict(te_d[0])]
    num_correct = sum(int(a ==y) for a,y in zip(predictions, te_d[1]))
    print("Baseline using SVM:")
    print("%s of %s values correct." % (num_correct,len(te_d[1])))

if __name__ == "__main__":
    svm_baseline()