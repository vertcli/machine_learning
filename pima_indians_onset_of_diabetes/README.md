Pima Indians Diabetes Classification
-----------------------------------

The aim of this project is to deal with a two class classification problem using the Pima Indians Diabetes dataset. This dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes) and it contains 768 samples and 8 features. This is a subset of a larger dataset and it only contains samples from females from at least 21 years old. This project will implement Logistic Regression to solve this classification problem and will tune the parameter C (C=1/lambda) to see who it performs and the way this parameter underfit or overfit our problem.

Requirements
----------------------
* You should have installed Anaconda on your system. If not, you can install it from [here](https://docs.continuum.io/anaconda/install).
* Make sure you use Python 2.

Installation
----------------------

### Create the environment and install the requirements
 
* Clone this repo to your computer by typing `git clone https://github.com/vertcli/machine_learning.git`.
* Get into the folder using `cd machine_learning/pima_indians_onset_of_diabetes`.
* Create a python environment using `conda create --name pima_indians_onset_of_diabetes --file requirements.txt python=2`.
    
### Download the data

* Activate this environment: `source activate pima_indians_onset_of_diabetes`.
* Run `mkdir data`.
* Run `python diabetes_data_reader.py`.
    * This will create `raw_diabetes.csv` in the `data` folder. The data will look like:
    ![alt tag](https://github.com/vertcli/machine_learning/blob/master/pima_indians_onset_of_diabetes/img/raw_data.jpg)

Usage
-----------------------

* If you are not already into pima_indians_onset_of_diabetes folder get into it. Also, make sure you have pima_indians_onset_of_diabetes environment active.
* Run `python diabetes_process_data.py` normalize and center the dataset.
    * This will create `processed_data.csv` in the `data` folder. The data will look like:
    ![alt tag](https://github.com/vertcli/machine_learning/blob/master/pima_indians_onset_of_diabetes/img/processed_data.jpg)    
* Run `python diabetes_pca.py`.
    * This will show up the variance of each feature in the new space. In this case our data has significant variance on several directions. 
    * A second plot will show the projection of the dataset over the two most relevant directions.
    * It will also export these plots into the `img` folder.
* OPTIONAL: Run `python diabetes_visualization.py`.
    * This will show up a pair-plot to see the correlation between the features of our dataset.
    * It will also export this plot into the `img` folder.
* Run `python diabetes_logistic_regression_classification.py`.
    * This will run logistic regression algorithm several times, each one with a different value of C. The dataset is divided into 10 folks and we do a cross validation across the folks. At the end, a plot shows the accuracy score vs C in a box-plot. One can observe that for low values of C (large values of lambda) we have underfitting and the accuracy is very low. On the other hand, for large values of C (low values of lambda) we reach a steady state where we have overfitting, but the accuracy is not increasing as C increase because there are some values which our logistic regression will never be able to classify. Note: this graph is also exported to the `img` folder.

Extending this
-------------------------

If you want to extend this work, here are a few places to start:

* You can explore different values of C editing the file `settings.py`. You will observe that for C>1000 the accuracy is around 77.4% and as small C is you get every time pourer results.
* You can try to create polinomial features by multiplying features. This is an easy way to create features, but the end result may not improve your algorithm.