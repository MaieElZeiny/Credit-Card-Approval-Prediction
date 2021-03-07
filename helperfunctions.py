#Importing Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go # it's like "plt" of matplot
import matplotlib.patches as mpatches
import seaborn as sns
import pandas_profiling as ppf



import plotly.offline as py
import plotly.tools as tls # It's useful to we get some tools of plotly
import warnings # This library will be used to ignore some warnings
import itertools
from collections import Counter # To do counter of some features
from heatmap import heatmap, corrplot
from time import time
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report




#defintions of functions to be moved to helper file


def plotDistPlot(col):
    """Flexibly plot a univariate distribution of observation"""
    sns.distplot(col)
    plt.show()

def remove_outliers(data, column):
    q75,q25 = np.percentile(data.loc[:,column],[75,25])
    intr_qr = q75-q25
    
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    
    data.loc[data[column] < min,column] = np.nan
    data.loc[data[column] > max,column] = np.nan
    
    return data

def train_predict(classifiers, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - classifiers: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    # evaluate each model in turn
    results = {}
    # Fit the learner to the training data 
    start = time() # Get start time
    learner = classifiers.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    # Calculate the training time
    results['train_time'] = end - start     
    # Get the predictions on the test set
    start = time() # Get start time
    predictions_test = classifiers.predict(X_test)
    predictions_train = classifiers.predict(X_train[:300])
    end = time() # Get end time
    # Calculate the total prediction time
    results['pred_time'] = end -start   
    # Compute accuracy on the first 300 training samples 
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    # Compute Recall on the the first 300 training samples
    results['F1score_train'] = f1_score(y_train[:300], predictions_train)
    # Compute Recall on the test set which is y_test
    results['F1score_test'] = f1_score(y_test, predictions_test)
    # Success
    print("{} trained on {} samples.".format(classifiers.__class__.__name__, sample_size))
    # Return the results
    return results




def evaluate(results):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
    """
    # Create figure
    fig, ax = plt.subplots(6, 1, figsize = (10,16))

    # Constants
    colors = ['#A00000','#00A0A0','#00A000','#F37A41','#FFD966','#1F3084']
    n_series = 5
    n_observations = 3
    x = np.arange(n_observations)
    # Determine bar widths
    width_cluster = 0.7
    width_bar = width_cluster/n_series
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()): 
        for j, metric in enumerate(['train_time', 'acc_train', 'f1score_train', 'pred_time', 'acc_test', 'f1score_test']):
            for i in np.arange(n_observations):  #0,1,2 Sample
                # Creative plot code
                x_positions = i+(width_bar*k)-width_cluster/2
                ax[j%6].bar(i+k*width_bar, results[learner][i][metric], width = width_bar,color = colors[k])
                ax[j%6].set_xticks([0.45, 1.45, 2.45])
                ax[j%6].set_xticklabels(["1%", "10%", "100%"])
                #ax[j%6].set_xlabel("Training Set Size")
                ax[j%6].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0].set_ylabel("Time (in seconds)",fontsize=10)
    ax[1].set_ylabel("Accuracy Score",fontsize=10)
    ax[2].set_ylabel("Fscore",fontsize=10)
    ax[3].set_ylabel("Time (in seconds)",fontsize=10)
    ax[4].set_ylabel("Accuracy Score",fontsize=10)
    ax[5].set_ylabel("Fscore Score",fontsize=10)
    
    # Add titles
    ax[0].set_title("Model Training")
    ax[1].set_title("Accuracy Score on Training Subset")
    ax[2].set_title("Fscore Score on Training Subset")
    ax[3].set_title("Model Predicting")
    ax[4].set_title("Accuracy Score on Testing Set")
    ax[5].set_title("Fscore score on Testing Set")


   # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
        plt.legend(handles = patches, loc='upper center', bbox_to_anchor=(0.5,7.5),
                    ncol = 3, fontsize = 'large')
    # Aesthetics
    plt.suptitle("Performance Metrics for The Supervised Learning Models", fontsize = 0, y = 1.10)
    #plt.subplots(constrained_layout=True)
    #plt.tight_layout()
    plt.subplots_adjust(top=1.9)
    plt.show()


def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = plt.figure(figsize = (9,5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#0050A1', \
          label = "Cumulative Feature Weight")
    plt.bar(np.arange(5), values, width = 0.6, align="center", color = '#A10000', \
          label = "Feature Weight")
    plt.xticks(np.arange(5), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)
    
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show()  


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.YlOrRd:
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def train_predict(classifiers, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    # evaluate each model in turn
    results = {}
    # Fit the learner to the training data 
    start = time() # Get start time
    learner = classifiers.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    # Calculate the training time
    results['train_time'] = end - start     
    # Get the predictions on the test set
    start = time() # Get start time
    predictions_test = classifiers.predict(X_test)
    predictions_train = classifiers.predict(X_train[:300])
    end = time() # Get end time
    # Calculate the total prediction time
    results['pred_time'] = end -start   
    # Compute accuracy on the first 300 training samples 
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    # Compute Recall on the the first 300 training samples
    results['recall_train'] = recall_score(y_train[:300], predictions_train)
    # Compute Recall on the test set which is y_test
    results['recall_test'] = recall_score(y_test, predictions_test)
    # Success
    print("{} trained on {} samples.".format(classifiers.__class__.__name__, sample_size))
    # Return the results
    return results
    
    
