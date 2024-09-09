import sys; args =  sys.argv[1:]
import math
import random
import re 
import pandas as pd 
import cv2 
import numpy as np
import matplotlib.pyplot as plt

def function(x): return 1/(1+(math.e**(-x)))
def derivative(x): return (1-x)*x
def dot(x,y): return sum([x[i]*y[i] for i in range(len(x))])

def backProp(x,ogweights,outputs):
    alpha = 0.1
    error = [[0 for a in n] for n in x]
    weights = [[weight for weight in og] for og in ogweights]
    outputLayer = [output - x[-2][a]*weights[-1][a] for a,output in enumerate(outputs)] #output error t-y
    lastError = [outputLayer[i]*ogweights[-1][i]*derivative(x[-2][i]) for i in range(len(outputs))] #(output error)*finalweight*derivative(xfinal)
    lastGrad = [outputLayer[i]*(x[-2][i]) for i in range(len(outputs))] #(output error)*finalweight*xfinal
    for i in range(len(weights[-1])):
        weights[-1][i] += alpha*lastGrad[i] #update the weight
        error[-1][i] = outputLayer[i] #update the output layer
        error[-2][i] = lastError[i] #update the last layer
    for layer in range(len(x)-3,-1,-1): # A1 B1 C1 A2 B2 C2
        for i in range(len(x[layer])): #leftmost index
            Ei = 0
            for j in range(len(x[layer+1])): 
                neggradient = x[layer][i]*error[layer+1][j]
                Ei +=weights[layer][j*len(x[layer])+i]*error[layer+1][j]
                weights[layer][j*len(x[layer])+i] += alpha * neggradient #update the weights (weight+alpha*gradient)
            error[layer][i] = Ei*derivative(x[layer][i])
    return weights

def ff(NN,weights):
    for layer in range(0,len(NN)-1):
        if layer == len(NN)-2:
            for i in range(len(NN[layer+1])):
                NN[layer+1][i] = weights[layer][i]*NN[layer][i]
        else:
            for i in range(len(NN[layer+1])):
                NN[layer+1][i] = function(dot(NN[layer],weights[layer][i*(len(NN[layer])):i*(len(NN[layer]))+len(NN[layer])]))
    return NN
        
    
def makeNN(inputs,weights,outputs):
    NN = [[] for i in range(len(weights))]
    NN[0] = [input for input in inputs]
    NN[1] = [0 for i in range(32)]
    NN[2] = [0 for i in range(len(outputs))]
    NN.append([0 for i in range(len(outputs))])
    return NN


def grfParse():
    import csv
    import random
    import numpy as np 
    inputs = []
    final = [[None, None] for _ in range(60000)] #inputs, outputs, weights

    with open('mnist_train.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        first = True
        i = 0
        for row in csv_reader:
            if first: #skip the header
                first = False
                continue
            inputs = [int(a)/255.0 for a in row[1:]]
            inputs.append(1) #add the bias
            final[i][0] = inputs  # inputs
            outputs = [1 if j == (int)(row[0]) else 0 for j in range(10)]
            final[i][1] = outputs  # outputs
            i += 1
    weights = [
                [random.uniform(-0.5,0.5) for a in range(785*32)],
                [random.uniform(-0.5,0.5) for a in range(10*32)],
                [random.uniform(-0.5,0.5) for a in range(len(outputs))]
            ]
    return final,weights

def validationParse():
    import csv
    import random
    import numpy as np 
    inputs = []
    final = [[None, None] for _ in range(10000)] #inputs, outputs, weights

    with open('mnist_test.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        first = True
        i = 0
        for row in csv_reader:
            if first: #skip the header
                first = False
                continue
            inputs = [int(a)/255.0 for a in row[1:]]
            inputs.append(1) #add the bias
            final[i][0] = inputs  # inputs
            outputs = [1 if j == (int)(row[0]) else 0 for j in range(10)]
            final[i][1] = outputs  # outputs
            i += 1
    with open("output.txt", "r") as file:
        line = file.readline()
        weights = [[] for a in range(3)]
        count = 0
        while line:
            print(count)
            line = line.strip() 
            temp_weights = line.split(" ") 
            weights[count] = ([float(a) for a in temp_weights]) 
            line = file.readline() 
            count+=1
    return final,weights


def main():
    global alpha
    alpha = 0.2
    NNlist,weights = grfParse()
    accuracy_history = []
    epochs = 8
    for i in range(epochs):
        error_count = 0
        for a,nn in enumerate(NNlist):
            inputs = nn[0]
            outputs = nn[1]
            NN = makeNN(inputs, weights, outputs)
            NN = ff(NN, weights)
            outputValue = max(NN[-1])
            if(outputs.index(1) != NN[-1].index(outputValue)):
                error_count+=1
            weights = backProp(NN, weights, outputs)
        accuracy = (len(NNlist)-error_count)/len(NNlist)
        accuracy_history.append(accuracy)
    
    temp = "Layer counts "
    for a in NN:
        temp+=str(len(a))+" "
    print(temp)
    with open('output.txt', 'w') as file:
        for a in weights:
            tempstr = ""
            for x in a:
                tempstr += str(x)+" "
            file.write(tempstr+"\n")  
    print("Training accuracy over epochs")
    print(accuracy_history)
    NNlist,weights = validationParse()
    error_count = 0
    for a,nn in enumerate(NNlist):
        inputs = nn[0]
        outputs = nn[1]
        NN = makeNN(inputs, weights, outputs)
        NN = ff(NN, weights)
        outputValue = max(NN[-1])
        if(outputs.index(1) != NN[-1].index(outputValue)):
            error_count+=1
    accuracy = (len(NNlist)-error_count)/len(NNlist)
    print("Validation accuracy: ")
    print(accuracy)
    x = [a+1 for a in range(epochs)]
    plt.plot(x,accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.show() 

if __name__ == "__main__":
    main()

