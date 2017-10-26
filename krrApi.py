import numpy as np
from numpy import*
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import time
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from pylab import scatter, show, legend, xlabel, ylabel  
from sklearn.metrics import r2_score


def splitDataset(dataset, attributeList, predictY, testNum=50):
    X = dataset[attributeList[:]]
    X = np.array(X)
    
    Y = dataset[[predictY]]
    Y = np.array(Y)
 
    XTrains = X[0:X.shape[0] - testNum, :]
    YTrains = Y[0:Y.shape[0] - testNum, :]
    
    XTests = X[XTrains.shape[0]:, :]
    realY = Y[YTrains.shape[0]:, :]
    return XTrains, XTests, YTrains, realY
    
# read dataset 
dataset = pd.read_csv("data_akbilgic.csv", header=0)

# transform csv into numpy array 
attributeList = ["ISE", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"]
yAttribute = "SP"
X = dataset[attributeList[:]]
X = np.array(X)
Y = dataset[[yAttribute]]
Y = np.array(Y)

# set up train and test
Xtr, Xte, Ytr, realY = splitDataset(dataset, attributeList, yAttribute)

# linear kernel ridge model selkected by grid search 
kr1 = GridSearchCV(KernelRidge(kernel='linear', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
kr1_fit = kr1.fit(Xtr,Ytr)
Ytr1 = kr1.predict(Xtr)
Paras1 = kr1.get_params(deep=True)
score1 = kr1.score(Xtr, Ytr)
print("Training dataset R^2: ")
print("R^2 on train dataset by Linear KRR: ",score1) 
                            
# polynomial kernel ridge model selkected by grid search 
kr2 = GridSearchCV(KernelRidge(kernel='poly', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
kr2.fit(Xtr,Ytr)
Ytr2 = kr2.predict(Xtr)
Paras2 = kr2.get_params(deep=True)
score2 = kr2.score(Xtr, Ytr)
print("R^2 on train dataset by Polynomial KRR: ",score2)                          
                              
# gaussian kernel ridge model selkected by grid search                               
kr3 = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
kr3.fit(Xtr,Ytr)
Ytr3 = kr3.predict(Xtr)
Paras3 = kr3.get_params(deep=True)
score3 = kr3.score(Xtr, Ytr)
print("R^2 on train dataset by Gaussian KRR: ",score3)


# models run on test set 
Ypr1 = kr1.predict(Xte)
test_score1 = kr1.score(Xte, realY)
print("Testing dataset R^2: ")
print("R^2 on test dataset by Linear KRR: ",test_score1) 

Ypr2 = kr2.predict(Xte)
test_score2 = kr2.score(Xte, realY)
print("R^2 on test dataset by Polynomial KRR: ",test_score2) 

Ypr3 = kr3.predict(Xte)
test_score3 = kr3.score(Xte, realY)
print("R^2 on test dataset by Gaussian KRR: ",test_score3) 

# scatter plot
size = realY.shape[0]
indexArr = np.zeros(size)
for i in range(size):
    indexArr[i] = i
 
scatter(indexArr, realY, marker='o', c='black')  
scatter(indexArr, Ypr1, marker='x', c='red') 
scatter(indexArr, Ypr2, marker='x', c='blue')  
scatter(indexArr, Ypr3, marker='x', c='green')   
xlabel('index')  
ylabel('Real Y and Predictions of Testing Dataset')  
plt.title('Self-Implemented Models') 
legend(['Real Y', 'LinearKrr Y','PolyKrr Y','GussianKrr Y'])  
show()

