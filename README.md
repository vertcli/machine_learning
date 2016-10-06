Machine Learning projects
=============================

My name is Albert Climent Bigas and this is my Machine Learning (ML) portfolio. Each folder belongs to a different ML project developed in python 2. If you are interested to run one of my projects you just have to clone this repo in your computer and follow the instructions of the README file you will find inside each project. The python environment manager I used is [Conda](http://conda.pydata.org/docs/intro.html) (from [Anaconda](https://docs.continuum.io/anaconda/)) and it is the one I suggest you to use. 
When you want to deal with a machine learning problem a best practice is to split the problem in small pieces and solve each one indiviually. The basic pieces our problem should have are:

![alt tag](https://github.com/vertcli/images/blob/master/ML_flow.jpg)	

In the first step we choose which variables can affect our problem and which ones may be irrelevant. These variables are known as features and each measurement is known as sample. Usualy, is best practice to process the raw data so each feature is normalized and has the same weight when training our Machine Learning algorithm. It is also recomended to plot it so we can see how our data is distributed. Finally, it can be also interesting to perform a PCA analysis to see how correlated our data is in this second phase of the problem.
At this point we know some caracteristics of our dataset (i.e. two classes, more than two classes, unknown classes, strongly correlated data, number of independent features, etc.) so it is time to look for a candidate algorithm or to try to create more features (i.e. polynomial features). To train the chosen algorithm it is recommended to split our dataset into different folds to do cross-validation and see how well our algorithm is performing. This cross validation will be as part of the test phase, when we validate if our algorithm is performing as desired. Most of the algorithms have parameters to tune, so you can train different models with different parameters and observe how it behave. Some graphs or figures may help you visualize the performance of the different algorithms and you can comment them. This will help you to understand what is really happening. You may also want to add to the project a small discussion of the furure work/outgoing lines so other people may help you to improve your results.

Projects
--------

This portfolio is made of several ML mini-projects with the aim to show my skills on this field. The first one is called Glass Identification Classification, which is based on the UCI dataset with the same name.



