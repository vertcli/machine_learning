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
import settings as st

# Set numpy print options:
np.set_printoptions(precision = 2)

print('Reading data...')
# Read the processed data:
pd_glass = pd.read_csv('data/processed_data.csv')

#print(pd_glass.head(5))
x = pd_glass.as_matrix(st.FEATURE_NAMES)
y = pd_glass.as_matrix(['Type'])

cols = ['n_estimators','accuracy']
pd_accuracy = pd.DataFrame(columns=cols)

# Define a vector values that contain all the 'c' parameters we are going to study:
values = st.N_ESTIMATORS

for index in values:
    print('Trainging the algorithm with ' + str(index) + ' estimators...')
    # Define MultiLayer Perceptron Classifier class:
    classifier = mc.OneVsRestClassifier(abc(n_estimators=index))
    
    #Define the average accuracy over the folds:
    accuracy = np.empty(st.N_ITERATIONS)
    # Repeat the process and get the mean and std of the accuracy of this algorithm:
    for i in range(st.N_ITERATIONS):
        # Define kFold:
        kf = cv.KFold(len(x), n_folds = st.N_FOLKS, shuffle = True)

        accuracy[i] = 0

        # Train the algorithm and cross validate it:
        for train_index, test_intex in kf:
            classifier.fit(x[train_index],y[train_index])
            predicted = classifier.predict(x[test_intex])

            # Add the accuracy of this field to the mean:
            accuracy[i] += met.accuracy_score(y[test_intex], predicted)

        # Compute the average over the all folds:
        accuracy[i] /= st.N_FOLKS

        # Add the accuracy of this training to the list of accuracies:
        data = np.array([['',cols[0],cols[1]],[len(pd_accuracy),index,accuracy[i]]])
        pd_accuracy = pd.concat([pd_accuracy,pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:])])

    mean_accuracy = np.mean(accuracy)*100
    std_accuracy = np.std(accuracy)*100
    print('Accuracy: ' + '%.1f' %(mean_accuracy) + unichr(177) + '%.1f' %(std_accuracy) + '%')
    
# Convert the parameter 'c' and the 'accuracies' into floats:
pd_accuracy = pd_accuracy.astype(float)

print('Showing accuracy results...')
# Plot the results:
ax = sns.boxplot(x='n_estimators', y='accuracy', data=pd_accuracy)

plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.title('ADA Boost classifier results')
figure = plt.gcf()
plt.show()

# Save the figre:
figure.savefig("img/adaboost_n_classifiers_vs_accuracy.png")

print('Done!')


