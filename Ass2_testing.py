import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import KFold

#Data Generation

def dataSetGeneration():
    X_train = np.linspace(0.,1.,201) # training set
    X_test = np.linspace(0.,1.,101) # training set

    # Student number: 400237438
    np.random.seed(4387)

    t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(201)
    t_test = np.sin(4*np.pi*X_test) + 0.3 * np.random.randn(101)
    
    return X_train, t_train, X_test, t_test


def dataReshape(X_train,t_train,X_test, t_test):
    # Changing all the matrices to column vector
    X_train_col = X_train.reshape(-1,1)
    t_train_col = t_train.reshape(-1,1)

    X_test_col = X_test.reshape(-1,1)
    t_test_col = t_test.reshape(-1,1)
    
    return X_train_col,t_train_col,X_test_col,t_test_col

def trueFunc(x):
    trueFx = np.sin(4 * np.pi * x)
    return trueFx

# training error and cross-validation error for all k-NN models

def errorCalc(N,predict,target):

    # training error = 1/N * SUM(w0+w1x1+w2x2+...+wNxN - t)^2 from i = 1 to N
    error = np.sum(np.square(predict - target))/N

    return error


def kNNPrediction(k, X_train, X_test, t_train):
    prediction=[]
    
    for x_test in X_test:
        # Calculate euclidean distances between each x_test point and all data points in X_train
        EuDistance = [np.linalg.norm(x_train - x_test) for x_train in X_train]

        # Sort data points by distance (smallest to largest) and get first K numbers of nearest neighbors
        N_distance = np.argsort(EuDistance,kind='stable')[:k]

        # Get the target values of the K nearest neighbors
        kNN = [t_train[i] for i in N_distance]
        # kNN = list(t_train[N_distance])

        # Calculate the prediction as the mean of the target values of the K neighbors
        prediction.append(np.mean(kNN))
            
    return prediction
        

# Cross-validation 

def crossValid_prediction(k, k_fold, x_train, t_train):
    
    CV_preidction_mat = []
    
    CV_error = 0.0
    sc = StandardScaler()
    # cross validation data split
    for train, test in k_fold.split(x_train):
        x_train_sp, x_test_sp = x_train[train], x_train[test]
        t_train_sp, t_test_sp = t_train[train], t_train[test]   
                
        x_train_sp = sc.fit_transform(x_train_sp)
        x_test_sp = sc.transform(x_test_sp)

        cv_prediction = kNNPrediction(k,x_train_sp, x_test_sp, t_train_sp)
        CV_preidction_mat.append(cv_prediction)
        
        CV_error += errorCalc(t_test_sp.shape[0],cv_prediction,t_test_sp)
        
    return CV_error/ k_fold.n_splits


def plotFigure(X_train, t_train, X_test, predictor):
    
    fig = plt.figure()

    # plotting true function
    plt.plot(X_train, trueFunc(X_train), color = "black", label = "trueFunc" )

    # plotting training data
    plt.plot(X_train, t_train, 'o', color = "blue", label = "Training Data" )

    # plotting predictor function
    plt.plot(X_test, predictor, color = "red", label = "kNN Prediction" )

    plt.legend(loc="best")
    plt.show()

   
def plotRMSE(K, name, train_error):
    fig = plt.figure()
    fig.suptitle(f'Error vs K: {name}') 

    # plotting error vs k
    plt.plot(K, train_error, 'o', color = "blue", label = name)
      
    plt.legend(loc="best")
    plt.show()


def main():
    X_train, t_train, X_test, t_test = dataSetGeneration()
    
    X_train_rs,t_train_rs,X_test_rs,t_test_rs = dataReshape(X_train, t_train, X_test, t_test)
    
    # Scale both data set
    sc = StandardScaler()
    x_train = sc.fit_transform(X_train_rs)
    x_test = sc.transform(X_test_rs)

    # 1<= k <= 60
    kVal = np.arange(1,61,1)

    
    Fold = 5
    k_Fold = KFold(n_splits=Fold)
    
    training_error = []
    CV_error = []
    
    print(t_train_rs.shape[0])
    
    for k in kVal: 
        predict_kNN = kNNPrediction(k, x_train, x_train, t_train)
        train_error = errorCalc(t_train_rs.shape[0],predict_kNN,t_train_rs)
        training_error.append(train_error)
        
        
        CV_error.append(crossValid_prediction(k, k_Fold, x_train,t_train))
        print(f"k={k}, trainig error = {np.sqrt(train_error)}, cross-validation error = {CV_error[k-1]}")
       
    
    plotRMSE(kVal, "Trainig Error", np.sqrt(training_error))
    plotRMSE(kVal, "Cross-Validation Error", CV_error)
    
    k_best = np.argmin(CV_error)+1
    best_kNN = kNNPrediction(k_best, x_test, x_test, t_test)
    
    print(f"The best k is {k_best}")
    
    plotFigure(X_train, t_train, X_test, best_kNN)


if __name__ == '__main__':
    main()