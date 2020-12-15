import numpy as np
import math
import sys
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import time

# Function to generate data set of 2 classes based on given data
def GenData(size):

    class0 = {
        "mean": [1, 0],
        "Sigma": [[1, 0.75],[0.75, 1]],
        "class": 0
    }

    class1 = {
        "mean": [0, 1.5],
        "Sigma": [[1, 0.75],[0.75, 1]],
        "class": 1
    }
    # Generating dataset Data
    a = np.random.multivariate_normal(class0["mean"], class0["Sigma"], int(size/2))
    b = np.random.multivariate_normal(class1["mean"], class1["Sigma"], int(size/2))

    # Adding classes
    a = np.append(a, np.full((int(size/2), 1), class0["class"]), 1)    
    b = np.append(b, np.full((int(size/2), 1), class1["class"]), 1)

    dataset = np.concatenate((a, b))        # Concatanation both the arrays
    dataset = np.append(np.full((size,1), 1), dataset, 1)     # Adding Bias Term

    return dataset

def sigmoid(weight, val):
    z = np.sum(weight*val[:3])
    try: 
        sig = (1/(1+math.exp(-z)))
        return sig
    except:
        return 0

def LogisticRegression(batchFlag, threshold, learning_rate, TrainData, TestData):
    
    if batchFlag:
        print ("\nType: Batch Processing with learning rate: " + str(learning_rate))
        weight = np.ones(3)
        start = time.time()
        weight = batch(threshold, learning_rate, weight, TrainData)

    else:
        print ("\nType: Online Processing with learning rate: " + str(learning_rate))
        weight = np.ones(3)
        start = time.time()
        weight = online(threshold, learning_rate, weight, TrainData)
    print ("Time(s): "+ str(round(time.time() - start, 3)))
    print ("Weights: " + str(weight))
    testData(weight, TestData, batchFlag, learning_rate)
    

def online(threshold, learning_rate, weight, TrainData):
    iterationCounter = 0
    Er=[]
    Ic=[]
    Ig=[]
    gradnorm=[]
    while(True):
        if iterationCounter == 100000:
            print ("Iteration(s): " + str(iterationCounter))
            plotError(Er,Ic,"Online")
            plotGrad(gradnorm,Ig,"Online")
            return weight
        iterationCounter += 1
        pError = 0
        
        for x in TrainData:
            o = sigmoid(weight, x)
            gradient = (-learning_rate * (o - x[3]) * x[:3])
            
            weight = weight + gradient

            try: 
                Error = - x[3]*math.log(o) 
            except:
                Error = 0
            try:
                Error = Error - (1-x[3])*math.log(1-o)
            except:
                Error = Error
            Er.append(Error)
            Ic.append(iterationCounter)

            if round(Error,6) == round(pError,6):
                print ("Iteration(s): " + str(iterationCounter))
                plotError(Er,Ic,"Online")
                plotGrad(gradnorm,Ig,"Online")
                return weight
            pError = Error


            gradnorm.append(math.sqrt(gradient[0]**2+gradient[1]**2+gradient[2]**2))
            Ig.append(iterationCounter)

            if abs(gradient[0]) <= threshold and abs(gradient[1]) <= threshold and abs(gradient[2]) <= threshold:
                print ("Iteration(s): " + str(iterationCounter))
                plotError(Er,Ic,"Online")
                plotGrad(gradnorm,Ig,"Online")
                return weight


def batch(threshold, learning_rate, weight, TrainData):
    size = len(TrainData)
    iterationCounter = 0
    pError = -1
    Er=[]
    Ic=[]
    Ig=[]
    gradnorm=[]
    while(True):
        if iterationCounter == 100000:
            print ("Iteration(s): " + str(iterationCounter))
            plotError(Er,Ic,"Batch")
            plotGrad(gradnorm,Ig,"Batch")
            return weight
        iterationCounter += 1
        sum_weights = np.zeros((3), dtype=float)

        Sigmoid = SigmoidNpArray(TrainData, weight)     
        sum_weights = (np.dot((Sigmoid - TrainData[:,3]),TrainData[:,:3]))
        gradient = (-learning_rate * (sum_weights/size))
        weight = weight + gradient

        Error = 0
        for i in range(0, len(TrainData)):

            try: 
                Error = Error - TrainData[i][3]*math.log(Sigmoid[i]) 
            except:
                Error = Error
            try:
                Error = Error - (1-TrainData[i][3])*math.log(1-Sigmoid[i])
            except:
                Error = Error
            Er.append(Error)
            Ic.append(iterationCounter)

        if round(Error,6) == round(pError,6):
            print ("Iteration(s): " + str(iterationCounter))
            plotError(Er,Ic,"Batch")
            plotGrad(gradnorm,Ig,"Batch")
            return weight
        pError = Error

        gradnorm.append(math.sqrt(gradient[0]**2+gradient[1]**2+gradient[2]**2))
        Ig.append(iterationCounter)
        # Checking for convergence of gradients
        if abs(gradient[0]) <= threshold and abs(gradient[1]) <= threshold and abs(gradient[2]) <= threshold:
            print ("Iteration(s): " + str(iterationCounter))
            print ("Error: " + str(Error))
            plotError(Er,Ic,"Batch")
            
            plotGrad(gradnorm,Ig,"Batch")                
            return weight

# Generic Function to test Weights generated with the testing Dataset and plotting the scatter plots and decision boundary  
def testData(weight, dataSet, batchFlag, learning_rate):
    final = ActivationSort(dataSet, SigmoidNpArray(dataSet, weight))
    accuracy = 100*(float(np.count_nonzero(np.equal(final[:,3], final[:,4])))/len(dataSet))
    print ("Accuracy = " + str(accuracy))

    red_patch = mpatches.Patch(color='red', label='Class 0')
    green_patch = mpatches.Patch(color='green', label='Class 1')
    plt.legend(handles=[red_patch,green_patch],loc=1)
    a=int(len(dataSet)/2)
    
    x=dataSet[:a+1]
    y=dataSet[:a+1]
    plt.scatter(x[:,1],y[:,2],alpha=0.5,c='r')
    x=dataSet[a+1:]
    y=dataSet[a+1:]
    plt.scatter(x[:,1],y[:,2],alpha=0.5,c='g')
    
    a=weight[0]
    b=weight[1]
    c=weight[2]
    
    x=np.arange(np.min(dataSet[:,1]-1),np.max(dataSet[:,1])+1)
    y=-(a+np.dot(b,x))/c
    
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def SigmoidNpArray(dataSet, weight):
    Z = (np.sum(dataSet[: , :3]*weight, axis = 1, dtype = float))   # Finding sum 'Z'
    Sigmoid = 1/(1+np.exp(-1*Z))
    return Sigmoid

def plotError(Error,iterationCounter,name):
    plt.plot(iterationCounter,Error)
    plt.title("Changes in Training Error for "+name+" training")
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.show()

def plotGrad(gradient,iterationCounter,name):
    plt.plot(iterationCounter,gradient)
    plt.title("Changes in Gradient norm for "+name+" training")
    plt.ylabel("Gradient")
    plt.xlabel("Iterations")
    plt.show()

def ActivationSort(dataSet, Sigmoid):
    dataSet = np.column_stack((dataSet, np.rint(Sigmoid), Sigmoid))
    return dataSet[dataSet[:,5].argsort()]

# Main    
def __main__(argv):
    # Setting Threshold for the gradients to converge
    threshold = 0.001

    size = 1000             # Static size 

    # Generating Training and Testing Dataset
    TrainData = GenData(size)
    TestData = GenData(int(size/2))


    for l in [0.001,0.01, 0.1, 1]:
        LogisticRegression(True, threshold, l, TrainData, TestData)
        LogisticRegression(False, threshold, l, TrainData, TestData)

__main__(sys.argv)