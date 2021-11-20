import numpy as np
import sklearn.linear_model as skl

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import helper functions
# from project 1
import functions
import Printerfunctions

# imports neural network
import NeuralNetClassification # for image classification
import NeuralNetRegression # for regression analysis

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets # for mnist data if needed

import warnings

def generateDataset(seed,n,N):
    print("Generating the FrankeFunction Dataset:",end="")

    # Using a seed to ensure that the random numbers are the same everytime we run
    # the code. Useful to debug and check our code.
    np.random.seed(seed)

    # Basic information for the FrankeFunction
    # The degree of the polynomial (number of features) is given by
    n = n
    # the number of datapoints
    N = N

    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    # Remember to add noise to function 
    z = functions.FrankeFunction(x, y) + 0.01*np.random.rand(N)

    X = functions.create_X(x, y, n=n)

    # split in training and test data
    # assumes 75% of data is training and 25% is test data
    X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

    # scaling the data
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)
    
    y_train -= np.mean(y_train)
    y_test -= np.mean(y_test)
    
    
    print(" Done\n")
    return X_train,X_test,y_train,y_test

# Generate training and test data for the MNIST dataset
def generateDatasetImages():
    #import matplotlib.pyplot as plt
    print("\nGenerating dataset from MNIST \n")
    # ensure the same random numbers appear every time
    np.random.seed(0)

    # download MNIST dataset
    digits = datasets.load_digits()

    # define inputs and labels
    inputs = digits.images
    labels = digits.target

    n_inputs = len(inputs)
    inputs = inputs.reshape(n_inputs, -1)

    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                    test_size=test_size)

    # to categorical turns our integer vector into a onehot representation
    #    one-hot in numpy
    Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)  
    return X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot

# Generates a dataset using Wisconsin Breast Cancer Data
def generateDatasetWBC():
    print("\nGenerating dataset from Wisconsin Breast Cancer Data\n")
    from sklearn.datasets import load_breast_cancer
    cancer=load_breast_cancer()      #Download breast cancer dataset

    inputs=cancer.data       
    outputs=cancer.target    
    labels=cancer.feature_names[0:30]

    """
    print('The content of the breast cancer dataset is:')      #Print information about the datasets
    print(labels)
    print('-------------------------')
    print("inputs =  " + str(inputs.shape))
    print("outputs =  " + str(outputs.shape))
    print("labels =  "+ str(labels.shape))
    """

    x=inputs      #Reassign the Feature and Label matrices to other variables
    y=outputs    
    # Generate training and testing datasets

    #Select features relevant to classification (texture,perimeter,compactness and symmetry) 
    #and add to input matrix

    temp1=np.reshape(x[:,1],(len(x[:,1]),1))
    temp2=np.reshape(x[:,2],(len(x[:,2]),1))
    X=np.hstack((temp1,temp2))      
    temp=np.reshape(x[:,5],(len(x[:,5]),1))
    X=np.hstack((X,temp))       
    temp=np.reshape(x[:,8],(len(x[:,8]),1))
    X=np.hstack((X,temp))       

    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)   #Split datasets into training and testing
    
    #Convert labels to categorical when using categorical cross entropy
    Y_train_onehot=to_categorical_numpy(Y_train)     
    Y_test_onehot=to_categorical_numpy(Y_test)

    del temp1,temp2,temp
    
    epochs = 1000
    batch_size = 100
    eta = 0.01
    lmbd = 0.01

    n_categories = 2
    hidden_neurons = [30,20,10]

    dnn2 = NeuralNetClassification.NeuralNetClassification(X_train, Y_train_onehot,eta=eta,lmbd=lmbd,epochs=epochs,batch_size=batch_size,
                    hidden_neurons=hidden_neurons,n_categories=n_categories)
    dnn2.train()
    test_predict = dnn2.predict(X_test)

    # accuracy score from scikit library
    training_predict = dnn2.predict(X_train)
    
    test_predict=dnn2.predict(X_test)#_probabilities(X_test)

    print("Accuracy score on training set: {:.4f}".format(accuracy_score(Y_train, training_predict)))    
    print("Accuracy score on test set: {:.4f}".format(accuracy_score(Y_test, test_predict)))    
    
    return X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot

# for some reason this     
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    return onehot_vector    


def part_a(X_train,X_test,y_train,y_test,N):
    print("Starting Project 2: part a")
    print("")
    betaValues = functions.OLS(X_train,y_train)
    ytilde = X_train @ betaValues
    ypredict = X_test @ betaValues
    
    print("Checking if my code works correctly:")
    print("My MSE is: {:.4f}".format(functions.MSE(y_test,ypredict)))
    print("My R2 score is: {:.4f}".format(functions.R2(y_test,ypredict)))
    
    print("\nTesting against sklearn OLS: ")
    clf = skl.LinearRegression(fit_intercept=False).fit(X_train, y_train)
    print("MSE after scaling: {:.4f}".format(mean_squared_error(clf.predict(X_test), y_test)))
    print("R2 score after scaling: {:.4f}".format(clf.score(X_test,y_test)))
    
    print("\n""\n")

    beta_original = np.random.randn(X_train[0].size)
    beta = np.copy(beta_original) # test all methods with same beta
    # eta above 1.0 seems to give overflow errors in beta
    eta = 0.05 #1.0 #0.01 #1.0 #0.01
    Niterations = 10000

    for iter in range(Niterations):
        gradient = (2.0/N)*X_train.T @ ((X_train @ beta)-y_train)
        beta -= eta*gradient

    ypredict = X_test @ beta
    
    print("Checking to see if my GradientDescent code works correctly:")
    print("My MSE is: {:.4f}".format(functions.MSE(y_test,ypredict)))
    print("My R2 score is: {:.4f}".format(functions.R2(y_test,ypredict)))    
 
    epochs = 10000#000
    mini_batch_size = 14#12 #12
    eta = 0.1#05 #1.0 #0.01
    lmbd=0.001

    beta=np.copy(beta_original)
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    mini_batch_sizes = [2,4,6,8,10,12]
    

    
    # R2 scores OLS and Ridge vs eta values
    printer = Printerfunctions.Printerfunctions()
    printer.partA1(X_train,y_train,X_test,y_test,beta,eta_vals,mini_batch_sizes,epochs,"OLS","OLS1",scaling=False)
    
    #printer = Printerfunctions.Printerfunctions()
    printer.partA1(X_train,y_train,X_test,y_test,beta,eta_vals,mini_batch_sizes,epochs,"Ridge","Ridge1",scaling=False)    

    # R2 scores OLS and Ridge vs mini batch size
    #printer = Printerfunctions.Printerfunctions()    
    printer.partA1(X_train,y_train,X_test,y_test,beta,eta_vals,mini_batch_sizes,epochs,"OLS","OLS2",scaling=False)
    
    #printer = Printerfunctions.Printerfunctions()    
    printer.partA1(X_train,y_train,X_test,y_test,beta,eta_vals,mini_batch_sizes,epochs,"Ridge","Ridge2",scaling=False)    

    # R2 scores OLS and Ridge vs epochs
    print("Calculating R2 values for epochs. This is very time consuming")
    epochs = np.linspace(1000,10000,num=10).astype(int)
    
    #printer = Printerfunctions.Printerfunctions()    
    printer.partA1(X_train,y_train,X_test,y_test,beta,eta_vals,mini_batch_sizes,epochs,"OLS","OLS3",scaling=False)
    
    #printer = Printerfunctions.Printerfunctions()    
    printer.partA1(X_train,y_train,X_test,y_test,beta,eta_vals,mini_batch_sizes,epochs,"Ridge","Ridge3",scaling=False) 
  
    R2_scores_test = np.zeros((len(eta_vals),len(mini_batch_sizes)))
    beta=np.copy(beta_original)
    epochs=10000
    print("\nCalculating R2 for eta vs mini batch size for OLS regression")
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(mini_batch_sizes):
            print("Calculating R2 using eta: ",eta," lambda: ",lmbd)        
            R2=functions.SGD(X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,lmbd,"OLS",True,False)
            #print("R2 score is: ",R2)
            R2_scores_test[i,j] = R2

    printer = Printerfunctions.Printerfunctions()
    printer.partA2(R2_scores_test)

    print("\nCalculating R2 for eta vs lambda for Ridge regression")
    mini_batch_size = 12
    beta=np.copy(beta_original)        
    R2_scores_test = np.zeros((len(eta_vals),len(lmbd_vals)))    
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            print("Calculating R2 using eta: ",eta," lambda: ",lmbd)
            R2=functions.SGD(X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,lmbd,"Ridge",True,False)
            #print("R2 score is: ",R2)
            R2_scores_test[i,j] = R2

    printer.partA3(R2_scores_test)    

    beta=beta_original
    epochs=10000
    mini_batch_size = 14#12 #12
    eta = 0.01#05 #1.0 #0.01
    lmbd=0.001
    # summarize results
    functions.SGD(X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,lmbd,"OLS",False,False)
    eta=0.1
    # scaling is susceptible to overflow depending on hyperparameters
    functions.SGD(X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,lmbd,"OLS",False,True)
    

    print("\n\nSummary of results:")
    epochs=10000
    eta=0.01
    mini_batch_size=12
    functions.SGD(X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,lmbd,"OLS",False,False)

    epochs=10000
    eta=0.01
    mini_batch_size=12
    functions.SGD(X_train,y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,lmbd,"Ridge",False,False)    


def part_b_and_c(X_train,X_test,y_train,y_test):
    print("\n""\n")

    print("Starting Project 2: part b and c")
    print("\n")

    epochs = 10000
    batch_size = 100

    # Depending on the number of hidden layers and neurons
    # RELU and LeakyRELU seem to be susceptible to overflow errors when 
    # calculating the gradient. Varies with #epochs and eta
    # Not sure if it is something wrong with my code that causes it or
    # if they just need the right parameters to function correctly
    # Left hidden neurons as two layers below to show that it works
    # for multiple layers.

    # multiple hidden layers are very time consuming for RELU and LeakyRELU
    # uses 1 layer instead
    #hidden_neurons = [30,28]
    
    # Grid search is time consuming. Using fewer neurons
    hidden_neurons = [20]
    n_categories = 1

    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)

    DNN_numpy = np.zeros((len(eta_vals),len(lmbd_vals)),dtype=object)
    import NeuralNetRegression
    
    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            print("Training neural network for regression for eta: ",eta," and lambda: ",lmbd)
            dnn = NeuralNetRegression.NeuralNetRegression(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories,activation_f="sigmoid")
            dnn.train()
            DNN_numpy[i][j] = dnn

    MSE_scores_test = np.zeros((len(eta_vals),len(lmbd_vals)))
    R2_scores_test = np.zeros((len(eta_vals),len(lmbd_vals)))

    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            dnn = DNN_numpy[i][j]
            MSE_test,R2_test = dnn.predict(X_test,y_test,X_train,y_train,printMe=False)
            MSE_scores_test[i][j] = MSE_test
            R2_scores_test[i][j] = R2_test

    printer = Printerfunctions.Printerfunctions()
    printer.partB1(R2_scores_test)
    printer.partB2(MSE_scores_test)
    

    eta = 0.0001
    lmbd = 0.01

    ######## Sigmoid
    print("Training NN using sigmoid activation function")
    dnnRegression = NeuralNetRegression.NeuralNetRegression(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories,activation_f="sigmoid")
    dnnRegression.train()
    dnnRegression.predict(X_test,y_test,X_train,y_train,printMe=True)

    ######## RELU
    print("Training NN using RELU activation function")    
    dnnRegressionRELU = NeuralNetRegression.NeuralNetRegression(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories,activation_f="RELU")
    dnnRegressionRELU.train()
    dnnRegressionRELU.predict(X_test,y_test,X_train,y_train,printMe=True)

    ######## LeakyRELU
    print("Training NN using LeakyRELU activation function")     
    dnnRegressionLeakyRELU = NeuralNetRegression.NeuralNetRegression(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories,activation_f="LeakyRELU")
    dnnRegressionLeakyRELU.train()
    dnnRegressionLeakyRELU.predict(X_test,y_test,X_train,y_train,printMe=True)

def part_d(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset):
    print("\nStarting Project 2: part d")

    epochs = 1000
    batch_size = 100
    if (dataset=="MNIST"):
        n_categories=10
    if (dataset=="WBC"):
        n_categories=2
    hidden_neurons = [30,20,10]    

    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)

    DNN_numpy = np.zeros((len(eta_vals),len(lmbd_vals)),dtype=object)
    import NeuralNetClassification
    
    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            #print(eta," ",lmbd)
            dnn = NeuralNetClassification.NeuralNetClassification(X_train,Y_train_onehot,eta=eta,lmbd=lmbd,epochs=epochs,batch_size=batch_size,
                hidden_neurons=hidden_neurons,n_categories=n_categories)
            dnn.train()
            DNN_numpy[i][j] = dnn
            #test_predict = dnn.predict(X_test)

    accuracy_scores_train = np.zeros((len(eta_vals),len(lmbd_vals)))
    accuracy_scores_test = np.zeros((len(eta_vals),len(lmbd_vals)))

    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            dnn = DNN_numpy[i][j]
            train_pred = dnn.predict(X_train) 
            test_pred = dnn.predict(X_test)
            accuracy_scores_train[i][j] = accuracy_score(Y_train, train_pred)
            accuracy_scores_test[i][j] = accuracy_score(Y_test, test_pred)
          
    printer = Printerfunctions.Printerfunctions()    
    printer.partD(accuracy_scores_train,accuracy_scores_test,dataset)


def part_e(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset):
    print("\n""\n")

    print("Starting Project 2: part e")
    print("\n")    
    
    print("Logistic regression using ",dataset," image data")

    print("\nUsing Scikit-Learn:")
    logreg = skl.LogisticRegression(random_state=1,verbose=0,max_iter=1E4,tol=1E-8)
    logreg.fit(X_train,Y_train)
    train_accuracy    = logreg.score(X_train,Y_train)
    test_accuracy     = logreg.score(X_test,Y_test)

    print("Accuracy of training data: ",train_accuracy)
    print("Accuracy of test data: ",test_accuracy)

    epochs = 1000
    mini_batch_size = 12

    scaling=False#True
    printing=False
    
    if (dataset=="MNIST"):
        # 10 categories
        beta = np.random.randn(X_train[0].size,10)
    if (dataset=="WBC"):
        # 2 categories
        beta = np.random.randn(X_train[0].size,2)    
    
    scaling=False
    printing=True
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    accuracy_scores_train = np.zeros((len(eta_vals),len(lmbd_vals)))
    accuracy_scores_test = np.zeros((len(eta_vals),len(lmbd_vals)))    

    print("\nGenerating accuracy scores for Logistic Regression")
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            acc_train,acc_test=functions.LogRegression(X_train,Y_train_onehot,Y_train,X_test,Y_test,epochs,\
                                mini_batch_size,eta,beta,lmbd,scaling,printing)
            accuracy_scores_train[i,j]= acc_train
            accuracy_scores_test[i,j]= acc_test
           
    printer = Printerfunctions.Printerfunctions()
    printer.partE(accuracy_scores_train,accuracy_scores_test,dataset)



if __name__ == '__main__':
    print("---------------------")
    print("Running main function")
    print("---------------------\n")
    warnings.filterwarnings("ignore")
    # The seed
    seed = 2021
    # The degree of the polynomial (number of features) is given by
    n = 6 # change the polynomial level
    # the number of datapoints
    N = 1000    

    X_train, X_test, y_train, y_test = generateDataset(seed,n,N)

    part_a(X_train,X_test,y_train,y_test,N)
    part_b_and_c(X_train,X_test,y_train,y_test)
    
    # Generates the dataset for classification
    # using the Wisconsin Breast Cancer dataset
    X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot = generateDatasetWBC()
    dataset="WBC"
    part_d(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset)    
    part_e(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset)    
    
    # Generates the dataset for classification
    # using the MNIST dataset
    X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot=generateDatasetImages()
    dataset = "MNIST"
    part_d(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset)
    part_e(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset)