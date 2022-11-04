import numpy as np
import pdb
from numpy import float64, ndarray
from typing import List, Tuple, Union
class NeuralNetwork:


    def __init__(self, LayerDims: List[int] = [], Activations: List[str] = []) -> None:
        self.params = {}
        self.cache = []
        self.Activations = Activations
        self.CostFunction = ''
        self.lamb = 0
        self.Gradients = {}
        self.LayerType = ['']
        self.HyperParam = {}
        self.InitializeParameters(LayerDims)
        self.CheckActivations()


    def InitializeParameters(self, LayerDims: List[int]) -> None:
        NumLayers = int(len(self.params)/2)
        for i in range(1, len(LayerDims)):
            self.params["W" + str(NumLayers+i)] = (np.sqrt(2/LayerDims[i - 1])*np.random.randn(LayerDims[i],LayerDims[i - 1]))
            self.params["b" + str(i+NumLayers)] = np.zeros((LayerDims[i], 1))
            self.LayerType.append('fc')



    def AddFCN(self,dims: List[int],Activations: List[str]) -> None:
        self.InitializeParameters(dims)
        for i in Activations:
            self.Activations.append(i)



    def CheckActivations(self) -> None:
        NumLayers = int(len(self.params)/2)
        while len(self.Activations) < NumLayers :
            self.Activations.append(None)


    @staticmethod
    def LinearForward(A_prev, W, b):
        Z = W.dot(A_prev) + b
        LinearCache = [A_prev, W, b]
        return Z, LinearCache


    def Activate(self, Z, n_layer=1):
        ActivationCache = [Z]
        Activation = None
        if (self.Activations[n_layer - 1]) == None:
            Activation = Z
        elif (self.Activations[n_layer - 1]).lower() == "relu":
            Activation = Z * (Z > 0)
        elif (self.Activations[n_layer - 1]).lower() == "tanh":
            Activation = np.tanh(Z)
        elif (self.Activations[n_layer - 1]).lower() == "sigmoid":
            Activation = 1 / (1 + np.exp(-Z))
        elif (self.Activations[n_layer - 1]).lower() == "softmax":
            Activation = np.exp(Z-np.max(Z))
            Activation = Activation/(Activation.sum(axis=0)+1e-10)
        return Activation, ActivationCache




    def Forward(self, net_input: ndarray) -> ndarray:
        self.cache = [] 
        A = net_input
        for i in range(1, int(len(self.params) / 2)):
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]
            Z = LinearCache = None
            if self.LayerType[i] == 'fc':
                Z, LinearCache = self.LinearForward(A, W, b)
            A, ActivationCache = self.Activate(Z, i)
            if  self.LayerType[i]=='conv':
                if  self.LayerType[i+1] == 'fc':
                    A = A.reshape((A.shape[1]*A.shape[2]*A.shape[3],A.shape[0]))
            self.cache.append([LinearCache, ActivationCache])

        # For Last Layer
        W = self.params["W" + str(int(len(self.params) / 2))]
        b = self.params["b" + str(int(len(self.params) / 2))]
        Z, LinearCache = self.LinearForward(A, W, b)
        if len(self.Activations) == len(self.params) / 2:
            A, ActivationCache = self.Activate(Z, len(self.Activations))
            self.cache.append([LinearCache, ActivationCache])
        else:
            A = Z
            self.cache.append([LinearCache, [None]])
        
        return A



    def MSELoss(self,Predictions: ndarray,mappings: ndarray) -> float64:
        self.CostFunction = 'MSELoss'
        loss = np.square(Predictions-mappings).mean()/2
        regularization_cost = 0
        if self.lamb != 0:
            for params in range(len(self.cache)):  
                regularization_cost = regularization_cost + np.sum(np.square(self.params['W'+str(params+1)]))
        regularization_cost = (self.lamb/(2*Predictions.shape[1]))*regularization_cost
        return loss + regularization_cost


    def CrossEntropyLoss(self,Predictions: ndarray,mappings: ndarray) -> float64:
        epsilon = 1e-8
        self.CostFunction = 'CrossEntropyLoss'
        loss = -(1/Predictions.shape[1])*np.sum( mappings*np.log(Predictions+epsilon) + (1-mappings)*np.log(1-Predictions+epsilon) )
        regularization_cost = 0
        if self.lamb != 0:
            for params in range(len(self.cache)):
                regularization_cost = regularization_cost + np.sum(np.square(self.params['W'+str(params+1)]))
        regularization_cost = (self.lamb/(2*Predictions.shape[1]))*regularization_cost
        return loss + regularization_cost


    
    def OutputBackward(self,Predictions: ndarray,mapping: ndarray) -> ndarray:
        dA = None
        cost = self.CostFunction
        if cost.lower() == 'crossentropyloss':
            dA =  -(np.divide(mapping, Predictions+1e-10) - np.divide(1 - mapping, 1 - Predictions+1e-10))
        elif cost.lower() == 'mseloss':   
            dA =  (Predictions-mapping)
        return dA
    


    def Deactivate(self,dA: ndarray,n_layer: int) -> Union[ndarray, int]:
        ActivationCache = self.cache[n_layer-1][1]
        dZ = ActivationCache[0]
        deact = None
        if self.Activations[n_layer - 1] == None:
            deact = 1
        elif (self.Activations[n_layer - 1]).lower() == "relu":
            deact = 1* (dZ>0)
        elif (self.Activations[n_layer - 1]).lower() == "tanh":
            deact = 1- np.square(dA)
        elif (self.Activations[n_layer - 1]).lower() == "sigmoid" or (self.Activations[n_layer - 1]).lower()=='softmax':
            s = 1/(1+np.exp(-dZ+1e-10))
            deact = s*(1-s)
        return deact
    
    def LinearBackward(self,dA: ndarray,n_layer: int) -> Tuple[ndarray, ndarray, ndarray]:
        batch_size = dA.shape[1]
        current_cache = self.cache[n_layer-1]
        LinearCache = current_cache[0]
        A_prev,W,b = LinearCache
        dZ = dA*self.Deactivate(dA,n_layer)
        dW = (1/batch_size)*dZ.dot(A_prev.T) + (self.lamb/batch_size)*self.params['W'+str(n_layer)]
        db = (1/batch_size)*np.sum(dZ,keepdims=True,axis=1)
        dA_prev = W.T.dot(dZ)
        assert(dA_prev.shape == A_prev.shape)
        assert(dW.shape == W.shape)
        assert(db.shape == b.shape)
        return dW,db,dA_prev
        
        

    def Backward(self,Predictions: ndarray,mappings: ndarray) -> None:
        layer_num = len(self.cache)
        doutput = self.OutputBackward(Predictions,mappings)
        self.Gradients['dW'+str(layer_num)],self.Gradients['db'+str(layer_num)],self.Gradients['dA'+str(layer_num-1)] = self.LinearBackward(doutput,layer_num)
        temp = self.LayerType
        self.LayerType = self.LayerType[1:]
        for l in reversed(range(layer_num-1)):
            dW,db,dA_prev = None,None,None
            if self.LayerType[l] == 'fc':
                dW,db,dA_prev = self.LinearBackward(self.Gradients['dA'+str(l+1)],l+1)
            self.Gradients['dW'+str(l+1)] = dW
            self.Gradients['db'+str(l+1)] = db
            self.Gradients['dA'+str(l)] = dA_prev
        
        self.LayerType = temp
    
 


