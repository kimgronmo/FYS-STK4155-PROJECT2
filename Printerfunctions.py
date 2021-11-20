

# import functions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import to check code
import sklearn.linear_model as skl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# imports linear regression from scikitlearn to 
# check my code against
from sklearn.metrics import mean_squared_error

# imports Ridge regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# imports functions for real world data
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
import seaborn as sns

import functions
from sklearn.metrics import accuracy_score
# Where figures and data files are saved..
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)
if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)
if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)
def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)
def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)
def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

class Printerfunctions():
    
    def __init__(self):
        print("\n##### Printing data to files: #####")
        print("Starting printer to generate data:\n")
    

    def partA1(self,X_train,y_train,X_test,y_test,beta,eta_values,mini_batch_sizes,epochs,method,fig,scaling):
        mini_batch_size=12
        epoch=10000
        eta=0.01
        lmbd=0.001
     
        
        if (fig=="OLS1"):
            R2_scores=np.zeros(len(eta_values))
        if (fig=="Ridge1"):
            R2_scores=np.zeros(len(eta_values))
        if (fig=="OLS2"):
            R2_scores=np.zeros(len(mini_batch_sizes))
        if (fig=="Ridge2"):
            R2_scores=np.zeros(len(mini_batch_sizes))
        if (fig=="OLS3"):
            R2_scores=np.zeros(len(epochs))            
        if (fig=="Ridge3"):
            R2_scores=np.zeros(len(epochs)) 
            
        
        beta_original=np.copy(beta)
        if (method=="OLS"):
            counter=0
            if (fig=="OLS1"):
                for eta in eta_values:
                    print("Calculating R2 value for eta: ",eta," using OLS")                 
                    beta=beta_original
                    R2_scores[counter]=functions.SGD(X_train,y_train,X_test,y_test,epoch,mini_batch_size,eta,beta,lmbd,"OLS",True,False)
                    counter+=1
            if (fig=="OLS2"):        
                for mb in mini_batch_sizes:
                    print("Calculating R2 value for mini batch size: ",mb," using OLS")                 
                    beta=beta_original
                    R2_scores[counter]=functions.SGD(X_train,y_train,X_test,y_test,epoch,mb,eta,beta,lmbd,"OLS",True,False)
                    counter+=1                    
            if (fig=="OLS3"):        
                for epoch in epochs:
                    print("Calculating R2 value for epoch: ",epoch," using OLS") 
                    beta=beta_original
                    R2_scores[counter]=functions.SGD(X_train,y_train,X_test,y_test,epoch,mini_batch_size,eta,beta,lmbd,"OLS",True,False)
                    counter+=1                       
      
        if (method=="Ridge"):        
            counter=0    
            if (fig=="Ridge1"):
                for eta in eta_values:
                    print("Calculating R2 value for eta: ",eta," using Ridge")                
                    beta=beta_original
                    R2_scores[counter]=functions.SGD(X_train,y_train,X_test,y_test,epoch,mini_batch_size,eta,beta,lmbd,"Ridge",True,False)
                    counter+=1                
            if (fig=="Ridge2"):        
                for mb in mini_batch_sizes:
                    print("Calculating R2 value for mini batch size: ",mb," using Ridge")                  
                    beta=beta_original
                    R2_scores[counter]=functions.SGD(X_train,y_train,X_test,y_test,epoch,mb,eta,beta,lmbd,"Ridge",True,False)
                    counter+=1     
            if (fig=="Ridge3"):        
                for epoch in epochs:
                    print("Calculating R2 value for epoch: ",epoch," using Ridge")                 
                    beta=beta_original
                    R2_scores[counter]=functions.SGD(X_train,y_train,X_test,y_test,epoch,mini_batch_size,eta,beta,lmbd,"Ridge",True,False)
                    counter+=1    

        if (fig=="OLS1"):
            plt.figure(1)
            plt.title("Part a) R2 score as a function of $\eta$ values OLS", fontsize = 10)    
            plt.xlabel(r"$\eta$ values", fontsize=10)
            plt.ylabel(r"R2 score", fontsize=10)
            plt.plot(eta_values, R2_scores, label = "R2 score")
            plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'parta1 plot R2 vs eta_valuesOLS.png') \
                        , transparent=True, bbox_inches='tight')
            plt.close()                           
                        
        if (fig=="Ridge1"):
            plt.figure(2)
            plt.title("Part a) R2 score as a function of $\eta$ values Ridge", fontsize = 10)    
            plt.xlabel(r"$\eta$ values", fontsize=10)
            plt.ylabel(r"R2 score", fontsize=10)
            plt.plot(eta_values, R2_scores, label = "R2 score")
            plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'parta1 plot R2 vs eta_valuesRidge.png') \
                        , transparent=True, bbox_inches='tight')
            plt.close()            
        if (fig=="OLS2"):
            plt.figure(3)
            plt.title("Part a) R2 score as a function of mini batch sizes OLS", fontsize = 10)    
            plt.xlabel(r"mini batch sizes", fontsize=10)
            plt.ylabel(r"R2 score", fontsize=10)
            plt.plot(mini_batch_sizes, R2_scores, label = "R2 score")
            plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'parta1 plot R2 vs mini_batch_sizesOLS.png') \
                        , transparent=True, bbox_inches='tight')
            plt.close()            
                        
        if (fig=="Ridge2"):
            plt.figure(4)
            plt.title("Part a) R2 score as a function of mini batch sizes Ridge", fontsize = 10)    
            plt.xlabel(r"mini batch sizes", fontsize=10)
            plt.ylabel(r"R2 score", fontsize=10)
            plt.plot(mini_batch_sizes, R2_scores, label = "R2 score")
            plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'parta1 plot R2 vs mini_batch_sizesRidge.png') \
                        , transparent=True, bbox_inches='tight')
            plt.close()   
            
        if (fig=="OLS3"):
            plt.figure(5)
            plt.title("Part a) R2 score as a function of the number of epochs OLS", fontsize = 10)    
            plt.xlabel(r"The number of epochs", fontsize=10)
            plt.ylabel(r"R2 score", fontsize=10)
            plt.plot(epochs, R2_scores, label = "R2 score")
            plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'parta1 plot R2 vs epochsOLS.png') \
                        , transparent=True, bbox_inches='tight')
            plt.close()     
            
        if (fig=="Ridge3"):
            plt.figure(6)
            plt.title("Part a) R2 score as a function of the number of epochs Ridge", fontsize = 10)    
            plt.xlabel(r"The number of epochs", fontsize=10)
            plt.ylabel(r"R2 score", fontsize=10)
            plt.plot(epochs, R2_scores, label = "R2 score")
            plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'parta1 plot R2 vs epochsRidge.png') \
                        , transparent=True, bbox_inches='tight')
            plt.close()   

    def partB1(self,R2_test):
        print("\nPrinting R2 scores for Neural Network Regression to file")    
        sns.set()
        fig, ax = plt.subplots(figsize = (10,10))
        sns.heatmap(R2_test, annot=True, ax=ax, cmap="viridis")
        ax.set_title("R2 scores for Neural Network Regression")
        ax.set_ylabel("$\eta$ from 1e-5 to 1e1")
        ax.set_xlabel("$\lambda$ from 1e-5 to 1e1")
        plt.savefig(os.path.join(os.path.dirname(__file__),'Results/FigureFiles','partB1_NNRegressionR2.png') \
                    ,transparent=True,bbox_inches='tight')

    def partB2(self,MSE_test):
        print("\nPrinting MSE scores for Neural Network Regression to file")    
        sns.set()
        fig, ax = plt.subplots(figsize = (10,10))
        sns.heatmap(MSE_test, annot=True, ax=ax, cmap="viridis")
        ax.set_title("MSE scores for Neural Network Regression")
        ax.set_ylabel("$\eta$ from 1e-5 to 1e1")
        ax.set_xlabel("$\lambda$ from 1e-5 to 1e1")
        plt.savefig(os.path.join(os.path.dirname(__file__),'Results/FigureFiles','partB2_NNRegressionMSE.png') \
                    ,transparent=True,bbox_inches='tight')


    def partA2(self,R2_test):
        print("\nPrinting R2 scores for OLS regression using sgd to file")    
        sns.set()
        fig, ax = plt.subplots(figsize = (10,10))
        sns.heatmap(R2_test, annot=True, ax=ax, cmap="viridis")
        ax.set_title("R2 scores for test data OLS regression using sgd")
        ax.set_ylabel("$\eta$ from 1e-5 to 1e1")
        ax.set_xlabel("mini batch size")
        plt.savefig(os.path.join(os.path.dirname(__file__),'Results/FigureFiles','parta1_testR2vsetaOLS.png') \
                    ,transparent=True,bbox_inches='tight')
                    
    def partA3(self,R2_test):
        print("\nPrinting R2 scores for Ridge regression using sgd to file")    
        sns.set()
        fig, ax = plt.subplots(figsize = (10,10))
        sns.heatmap(R2_test, annot=True, ax=ax, cmap="viridis")
        ax.set_title("R2 scores for test data Ridge regression using sgd")
        ax.set_ylabel("$\eta$ from 1e-5 to 1e1")
        ax.set_xlabel("$\lambda$ from 1e-5 to 1e1")
        plt.savefig(os.path.join(os.path.dirname(__file__),'Results/FigureFiles','parta2_testR2vsetaRidge.png') \
                    ,transparent=True,bbox_inches='tight')                    

    def partD(self,train_accuracy,test_accuracy,dataset):
        print("\nPrinting results for dataset ",dataset," to file")    
        sns.set()
        fig, ax = plt.subplots(figsize = (10,10))
        sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Training Accuracy %s"%dataset)
        ax.set_ylabel("$\eta$ from 1e-5 to 1e1")
        ax.set_xlabel("$\lambda$ from 1e-5 to 1e1")
        plt.savefig(os.path.join(os.path.dirname(__file__),'Results/FigureFiles','training acc part d %s.png'%dataset) \
                    ,transparent=True,bbox_inches='tight')

        fig, ax = plt.subplots(figsize = (10,10))
        sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Test Accuracy %s"%dataset)
        ax.set_ylabel("$\eta$ from 1e-5 to 1e1")
        ax.set_xlabel("$\lambda$ from 1e-5 to 1e1")
        plt.savefig(os.path.join(os.path.dirname(__file__),'Results/FigureFiles','test acc part d %s.png'%dataset) \
                    ,transparent=True,bbox_inches='tight')


    def partE(self,accuracy_scores_train,accuracy_scores_test,dataset):
        print("\nPrinting results for dataset ",dataset," to file")
        sns.set()
        
        fig,ax = plt.subplots(figsize=(10,10))
        sns.heatmap(accuracy_scores_train, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Training Accuracy %s"%dataset)
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.savefig("Results/FigureFiles/accuracy_train_%s.png"%dataset)

        fig,ax = plt.subplots(figsize=(10,10))
        sns.heatmap(accuracy_scores_test, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Test Accuracy %s"%dataset)
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.savefig("Results/FigureFiles/accuracy_test_%s.png"%dataset)