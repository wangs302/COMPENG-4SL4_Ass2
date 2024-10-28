import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

#Data Generation

def dataSetGeneration():
    X_train = np.linspace(0.,1.,201) # training set
    X_test = np.linspace(0.,1.,101) # test set

    # Student number: 400237438
    np.random.seed(4387)

    #Target value generation
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
    #True function with no noise
    trueFx = np.sin(4 * np.pi * x)
    return trueFx

# training error and cross-validation error for all k-NN models

def errorCalc(N,predict,target):

    # training error = 1/N * SUM(y-t)^2 from i = 1 to N
    error = np.sum(np.square(predict - target))/N

    return error

# K nearest neighbours 
# Calculate euclidean distance, sort distance, get neighbours, average target values
def kNNPrediction(k, X_train, X_test, t_train):
    prediction=[]
    
    for x_test in X_test:
        # Calculate euclidean distances between each X_test point and all data points in X_train
        #The two dataset names are not representative for their actual use
        EuDistance = [np.linalg.norm(x_train - x_test) for x_train in X_train]

        # Sort data points by distance (smallest to largest) and get first K numbers of nearest neighbors
        N_distance = np.argsort(EuDistance,kind='stable')[:k]

        # Get the target values of the K nearest neighbors
        kNN = [t_train[i] for i in N_distance]

        # Calculate the prediction as the mean of the target values of the K nearest neighbors
        prediction.append(np.mean(kNN))
            
    return prediction
        

# Cross-validation 
# Split dataset into k equal parts, take k - 1 sets as training set, and 1 remaining set as test set 
# Train KNN based on each combination
def crossValid_prediction(k, k_fold, x_train, t_train):

    CV_error = 0.0
    sc = StandardScaler()
    
    # cross validation data split using KFold
    for train, test in k_fold.split(x_train):
        x_train_sp, x_test_sp = x_train[train], x_train[test]
        t_train_sp, t_test_sp = t_train[train], t_train[test]   
                
        # rescale dataset
        x_train_sp = sc.fit_transform(x_train_sp)
        x_test_sp = sc.transform(x_test_sp)
        
        # kNN prediction on each training and test set combination
        cv_prediction = kNNPrediction(k,x_train_sp, x_test_sp, t_train_sp)       
        
        # Sum up training error for each dataset combination
        CV_error += errorCalc(t_test_sp.shape[0],cv_prediction,t_test_sp)
        
    return CV_error/ k_fold.n_splits


def plotFigure(X_train, t_train, X_test, predictor):
    
    fig = plt.figure()

    # plotting true function
    plt.plot(X_train, trueFunc(X_train), color = "black", label = "trueFunc" )

    # plotting training data or test data
    plt.plot(X_train, t_train, 'o', color = "blue", label = "Training Data" )

    # plotting predictor function
    plt.plot(X_test, predictor, color = "red", label = "kNN Prediction" )

    plt.legend(loc="best")
    plt.show()

   
def plotError(K, name, error):
    fig = plt.figure()
    fig.suptitle(f'Error vs K: {name}') 

    # plotting error vs k
    plt.plot(K, error, 'o', color = "blue", label = name)
      
    plt.legend(loc="best")
    plt.show()
    

def errorCompare(K, train_error, cv_error):
    fig = plt.figure()
    fig.suptitle(f'Error vs K') 

    # plotting both training error and cross-validation error vs K on the same graph
    plt.plot(K, train_error, 'o', color = "blue", label = "Training Error")
    plt.plot(K, cv_error, 'o', color = "green", label = "Cross-Validation Error")
    
    plt.legend(loc="best")
    plt.show()

def main():
    X_train, t_train, X_test, t_test = dataSetGeneration() # Generate data
    
    X_train_rs,t_train_rs,X_test_rs,t_test_rs = dataReshape(X_train, t_train, X_test, t_test) #Reshape Data
    
    # Scale both data set
    sc = StandardScaler()
    x_train = sc.fit_transform(X_train_rs)
    x_test = sc.transform(X_test_rs)

    # 1<= k <= 60
    kVal = np.arange(1,61,1)

    # k fold where k = 5
    Fold = 5
    k_Fold = KFold(n_splits=Fold)
    
    # Documenting errors 
    training_error = []
    CV_error = []
    
    # Running throuhg all k values
    for k in kVal: 
        
        # Run basic kNN prediction with a specific k value
        predict_kNN = kNNPrediction(k, x_train, x_train, t_train)
        #Calculate training error
        train_error = errorCalc(t_train_rs.shape[0],predict_kNN,t_train_rs.flatten())
        training_error.append(train_error)
        
        # Run cross validation prediction and obtain cross-validation error
        CV_error.append(crossValid_prediction(k, k_Fold, x_train,t_train))
        print(f"k={k}, training error = {np.sqrt(train_error)}, cross-validation error = {CV_error[k-1]}")
       
    # PLot training error and cross-validation error
    plotError(kVal, "Training Error", np.sqrt(training_error))
    plotError(kVal, "Cross-Validation Error", CV_error)
    errorCompare(kVal, np.sqrt(training_error), CV_error)
    
    # Choose best k by choosing the smallest cross-validation error
    k_best = np.argmin(CV_error)+1
    # Test set performance
    best_kNN = kNNPrediction(k_best, x_test, x_test, t_test)
    # Test set error calcualtion
    test_error = errorCalc(t_test_rs.shape[0],best_kNN,t_test_rs.flatten())
    
    print(f"The best k is {k_best}, the test error is {np.sqrt(test_error)}")
    
    #Plotting the best k prediction with test set
    plotFigure(X_test, t_test, X_test, best_kNN)


if __name__ == '__main__':
    main()