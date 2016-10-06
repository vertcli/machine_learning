Machine Learning projects
=============================

My name is Albert Climent Bigas and this is my Machine Learning (ML) portfolio. Each folder belongs to a different ML project developed in python 2. If you are interested to run one of my projects you just have to clone this repo in your computer and follow the instructions of the README file you will find inside each project. The python environment manager I used is [Conda](http://conda.pydata.org/docs/intro.html) (from [Anaconda](https://docs.continuum.io/anaconda/)) and it is the one I suggest you to use. 
When you want to deal with a machine learning problem a best practice is to split the problem in small pieces and solve each one indiviually. The basic pieces our problem should have are:

![alt tag](https://github.com/vertcli/images/ML_flow.jpg)	

In the first step we choose which variables can affect our problem and which ones may be irrelevant. Ones the set of variables (features) are known we capture our raw data. Usualy, this data can not be processed directly by our Machine Learning algorithms and have to be preprocessed in a data processing step (i.e. normalize, center, etc.). It is also recomended to plot it so we a first visualization of the dataset. It can be interesting to perform a PCA analysis to see how correlated our data is.
After this step is performed, we have to train our learning algorithm. We usualy split our dataset into different folds to do cross-validation and see how well our algorithm is performing. All this information should then be summarized in a small document, in which you may also want to include some images or graphs of the results. You may also want to include a small discussion of the furure work/outgoing lines.

The Project
===========

In this problem we try to solve a multiclass classification problem. The dataset is made with 214 samples and 9 different features. The dataset looks like:



