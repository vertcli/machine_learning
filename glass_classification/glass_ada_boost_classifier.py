#######################################################
# Author: Albert Climent Bigas (vert.cli@gmail.com)
# Description: ML script using Adaptive Boosting algorithm to 
# solve Glass Identification dataset.
######################################################

# Import the required libraries:
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn import multiclass as mc
from sklearn import cross_validation as cv
from sklearn import metrics as met
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import unicodedata
import pandas as pd

# Set numpy print options:
np.set_printoptions(precision = 2)

# Read the processed data:
glass = np.genfromtxt('processed/data.csv', delimiter=',')

x = np.delete(glass,9,1)
y = glass[:,9]
n = 10            # Number of iterations
k = 10            # Number of folks

cols = ['n_estimators','accuracy']
pd_accuracy = pd.DataFrame(columns=cols)

# Define a vector values that contain all the 'c' parameters we are going to study:
values = np.array([10,40,80])

for index in values:
    # Define MultiLayer Perceptron Classifier class:
    classifier = mc.OneVsRestClassifier(abc(n_estimators=index))
    
    #Define the average accuracy over the folds:
    accuracy = np.empty(n)
    # Repeat the process and get the mean and std of the accuracy of this algorithm:
    for i in range(n):
        # Define kFold:
        kf = cv.KFold(len(x), n_folds = k, shuffle = True)

        accuracy[i] = 0

        # Train the algorithm and cross validate it:
        for train_index, test_intex in kf:
            classifier.fit(x[train_index],y[train_index])
            predicted = classifier.predict(x[test_intex])

            # Add the accuracy of this field to the mean:
            accuracy[i] += met.accuracy_score(y[test_intex], predicted)

        # Compute the average over the all folds:
        accuracy[i] /=k

        # Add the accuracy of this training to the list of accuracies:
        data = np.array([['',cols[0],cols[1]],[len(pd_accuracy),index,accuracy[i]]])
        pd_accuracy = pd.concat([pd_accuracy,pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:])])

    mean_accuracy = np.mean(accuracy)*100
    std_accuracy = np.std(accuracy)*100
    print('Accuracy: ' + '%.1f' %(mean_accuracy) + unichr(177) + '%.1f' %(std_accuracy) + '%')
    
# Convert the parameter 'c' and the 'accuracies' into floats:
pd_accuracy = pd_accuracy.astype(float)

# Plot the results:
ax = sns.boxplot(x='n_estimators', y='accuracy', data=pd_accuracy)
plt.show()

# Save the figre:
plt.figure().savefig("img/adaboost_n_classifiers_vs_accuracy.png")



