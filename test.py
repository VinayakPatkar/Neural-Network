import sklearn.datasets as datasets
import NeuralNetwork
import numpy as np
from optimizer import Optimizers
import pandas as pd
def runtestcases():
    print('Running test cases')
    np.random.seed(3)
    print('Running a binary classification test')

    #Generate sample binary classification data
    data = datasets.make_classification(n_samples=30000,n_features=10,n_classes=2)
    X= data[0].T
    Y = (data[1].reshape(30000,1)).T
    NeuralNet=NeuralNetwork.NeuralNetwork([10,25,1],['relu','sigmoid']);
    print(NeuralNet)
    NeuralNet.costfunction='CrossEntropyLoss'
    optim=Optimizers.SGDOptimizer
    optim(X,Y,NeuralNet,128,alpha=0.07,epoch=10,lamb=0.05,printstatement=1)
    output=NeuralNet.Forward(X)
    output = 1*(output>=0.5)
    accuracy = np.sum(output==Y)/30000
    print('for sgd without momentum\n accuracy = ' ,accuracy*100)
if __name__ == "__main__":
    runtestcases()