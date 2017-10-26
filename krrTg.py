import numpy as np
import pandas as pd
from numpy import*
from pylab import scatter, show, legend, xlabel, ylabel  
from sklearn.metrics import r2_score

def readFile(path):
    dataset = pd.read_csv(path, header=0)
    return dataset

def splitDataset(dataset, attributeList, predictY, testNum=10):
    X = dataset[attributeList[:]]
    X = np.array(X)
    
    Y = dataset[[predictY]]
    Y = np.array(Y)
 
    XTrains = X[0:X.shape[0] - testNum, :]
    YTrains = Y[0:Y.shape[0] - testNum, :]
    
    XTests = X[XTrains.shape[0]:, :]
    realY = Y[YTrains.shape[0]:, :]
    return XTrains, XTests, YTrains, realY

def kernel(x, X, kernelType, r=1, M=2, gamma=1, sigma=0.5):
    if kernelType == "linear":
        mKernel = np.dot(x, X.T)
    elif kernelType == "polynomial":
        mKernel = (gamma*np.dot(x, X.T) + r) ** M
    elif kernelType == "gaussian":
        rowx = x.shape[0]
        rowX = X.shape[0]
        mKernel = np.zeros((rowx, rowX))
        for i in range(rowx):
            for j in range(rowX):
                temp = x[i][:] - X[j][:]
                mKernel[i,j] = exp(- np.dot(temp,temp) / (2 * sigma ** 2) )
    else:
        return None
    return mKernel

def kCrossValidation(x, X, Y, k = 5, kernelType="linear", lambda_=1, r=2, M=2, gamma=1, sigma=0.5):
    # doing five cross validation
    size = X.shape[0]
    for i in range(k):
        XvalidationSet = X[size/k*i:size/k*(i+1)][:]
        xvalidationSet = x[size/k*i:size/k*(i+1)][:]
        YvalidationSet = Y[size/k*i:size/k*(i+1)][:]

def predict(x, X, Y, kernelType="linear", lambda_=1, r=2, M=2, gamma=1, sigma=0.5):
    try:
        K = kernel(X, X, kernelType, r, M, gamma, sigma)
        k = kernel(x, X, kernelType, r, M, gamma, sigma)
        I = np.eye(K.shape[0])
        beta = mat(K + lambda_ * I).I * Y
        predictY = k * beta
    except BaseException as err:
        print(format(err))
    else:
        return predictY
    
def RSquare(yHat, realY):
    yBar = sum(realY) / realY.shape[0]
    SSR = np.dot( (yHat - yBar).T, yHat - yBar)
    SST = np.dot( (realY - yBar).T , realY - yBar )
    SSE = np.dot( (realY - yHat).T , realY - yHat)
    temp = 1 - SSE / SST
    rSquare = temp[0, 0]
    if rSquare <= 0:
        return 0
    else:        
        return rSquare
    
# attributeList: 1-D string list
# yAttribute: string
def main(path, attributeList, yAttribute):
    dataset = readFile(path)
    XTrains, XTests, YTrains, realY = splitDataset(dataset, attributeList, yAttribute)

    predictY = predict(XTests, XTrains, YTrains, kernelType="gaussian", lambda_=0.3, sigma=1)
    rSquare = RSquare(predictY, realY)
    print("Gaussian kernel R^2 = " + str(rSquare))


# attributeList = [ "DAX", "FTSE", "BOVESPA", "EU", "EM"]
# yAttribute = "SP"

attributeList = [ "x1", "x2" ]
yAttribute = "y"


main("trivial_gussian.csv", attributeList, yAttribute)



