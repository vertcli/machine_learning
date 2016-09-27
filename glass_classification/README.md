Project: Glass Identification
=============================

When you want to deal with a machine learning problem you have to split the problem in small pieces. The basic pieces our problem should have are:

Data collection			->			Data processing				->		Machine learning phase			-> 			Summary	

In the first step we choose which variables can affect our problem and which ones may be irrelevant. Ones the set of variables (features) are known we capture our raw data. Usualy, this data can not be processed directly by our Machine Learning algorithms and have to be preprocessed in a data processing step (i.e. normalize, center, etc.). It is also recomended to plot it so we a first visualization of the dataset. It can be interesting to perform a PCA analysis to see how correlated our data is.
After this step is performed, we have to train our learning algorithm. We usualy split our dataset into different folds to do cross-validation and see how well our algorithm is performing. All this information should then be summarized in a small document, in which you may also want to include some images or graphs of the results. You may also want to include a small discussion of the furure work/outgoing lines.

The Project
===========

In this problem we try to solve a multiclass classification problem. The dataset is made with 214 samples and 9 different features. The dataset looks like:

![alt tag](https://github.com/vertcli/machine_learning/blob/master/glass_classification/img/head.jpg)

We see that not all features are normalized, so the first step is to normalize and center them by applying:

x = (x-sum(x)/n)/(max(x)-min(x))

The resulted dataset looks like:

![alt tag](https://github.com/vertcli/machine_learning/blob/master/glass_classification/img/normalized_head.jpg)

In this way, each feature will have the same weight on the learning process.

