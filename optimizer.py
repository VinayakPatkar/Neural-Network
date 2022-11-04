import NeuralNetwork
import numpy as np
import pandas as pd
class Optimizers:
    @staticmethod
    def GradientDescentOptimizer(input,realmapping,NeuralNetwork,alpha=0.001,lamb=0,epoch=100,printstatement=5,printstatus=True,update=True):
        NeuralNetwork.lamb=lamb;
        for i in range(epoch):
            NeuralNetwork.cache=[];
            #predictions by performing the Forward pass on the NeuralNetwork using the input data
            predictions=NeuralNetwork.Forward(input);
            #Identifying the loss function to be used
            lossfunction=(NeuralNetwork.costfunction).lower();
            loss=None
            r_cost=0
            if lossfunction=='mse':
                loss=NeuralNetwork.MSELOSS(predictions,realmapping);
            if lossfunction=='crossentropyloss':
                loss=NeuralNetwork.CrossEntropyLoss(predictions,realmapping);
            if printstatus and i%printstatement==0:
                print('Epoch :',i,'Loss:',loss);
            NeuralNetwork.Backward(predictions,realmapping);
            if update:
                NeuralNetwork.params=Optimizers.updateparams(NeuralNetwork.params,NeuralNetwork.Gradients,alpha)


    @staticmethod
    def SGDOptimizer(input,realmapping,NeuralNetwork,minbatchsize=64,alpha=0.001,lamb=0,momentum=None,epoch=5,printstatement=5,printstatus=True):
        batch_size = input.shape[1]
        mini_batches = []
        permutation = list(np.random.permutation(batch_size))
        shuffled_input = input[:,permutation]
        shuffled_mappings = (realmapping[:,permutation])

        num_complete_batches = int(np.floor(batch_size/minbatchsize))
        
        #Separate the complete mini_batches
        for i in range(0,num_complete_batches):
            mini_batch_input = shuffled_input[:,i*minbatchsize:(i+1)*minbatchsize]
            mini_batch_mappings = shuffled_mappings[:,i*minbatchsize:(i+1)*minbatchsize]
            mini_batch = (mini_batch_input,mini_batch_mappings)
            mini_batches.append(mini_batch)
        
        #Separate the incomplete mini batch if any
        if batch_size % minbatchsize != 0:
            mini_batch_input = shuffled_input[:,batch_size - num_complete_batches*minbatchsize : batch_size]
            mini_batch_mappings = shuffled_mappings[:,batch_size - num_complete_batches*minbatchsize : batch_size]
            mini_batch = (mini_batch_input,mini_batch_mappings)
            mini_batches.append(mini_batch)
        
        #Initialize momentum velocity
        velocity = {}
        if momentum != None:
            for i in range(int(len(NeuralNetwork.parameters)/2)):
                velocity['dW'+str(i+1)] = np.zeros(NeuralNetwork.parameters['W'+str(i+1)].shape)
                velocity['db'+str(i+1)] = np.zeros(NeuralNetwork.parameters['b'+str(i+1)].shape)
        

        for i in range(1,epoch+1):

            for batches in range(len(mini_batches)):

                if momentum != None:
                    Optimizers.GradientDescentOptimizer(input,realmapping,NeuralNetwork,alpha,lamb,epoch=1,printstatus=False,update=False)
                    for j in range(int(len(NeuralNetwork.parameters)/2)):
                        velocity['dW' + str(j+1)] = momentum*velocity['dW'+str(j+1)] + (1-momentum)*NeuralNetwork.Gradients['dW'+str(j+1)]
                        velocity['db' + str(j+1)] = momentum*velocity['db'+str(j+1)] + (1-momentum)*NeuralNetwork.Gradients['db'+str(j+1)]
                    NeuralNetwork.parameters = Optimizers.update_params(NeuralNetwork.parameters,velocity,alpha)
                else:
                    Optimizers.GradientDescentOptimizer(input,realmapping,NeuralNetwork,alpha,lamb,epoch=1,printstatus=False)

            prediction = NeuralNetwork.Forward(input)
            loss = None 
            loss_function = NeuralNetwork.CostFunction.lower()
            if loss_function == 'mseloss':
                loss = NeuralNetwork.MSELoss(prediction,realmapping)
            if loss_function == 'crossentropyloss':
                loss = NeuralNetwork.CrossEntropyLoss(prediction,realmapping)
            
            if i%printstatement == 0:
                print('Epoch :',i,'Loss:',loss);
    

    @staticmethod
    def updateparams(params,updation,learning_rate):
        '''
        Updates the parameters using gradients and learning rate provided
        
        :param params   : Parameters of the NeuralNetworkwork
        :param updation    : updation valcues calculated using appropriate algorithms
        :param learning_rate: Learning rate for the updation of values in params

        :return : Updated params 
        '''
        
        for i in range(int(len(params)/2)):
            params['W' + str(i+1)] = params['W' + str(i+1)] - learning_rate*updation['dW' + str(i+1)]
            params['b' + str(i+1)] = params['b' + str(i+1)] - learning_rate*updation['db' + str(i+1)]
        
        return params