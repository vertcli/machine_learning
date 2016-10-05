Glass Identification Classification
-----------------------------------

The aim of this project is to deal with a multiclass classification problem using glass information. After training the algorithm we will predict which type of glass each sample have. The dataset used for this problem is public and can be found [here](https://archive.ics.uci.edu/ml/datasets/Glass+Identification) and it contains 214 samples and 9 features. As you can read in this link, the study of glass classification was motivated by criminological investigations (i.e. determining the type of glass left on a scene of a crime can be used as evidence).

Requirements
----------------------
* You should have installed Anaconda on your system. If not, you can install it from [here](https://docs.continuum.io/anaconda/install).
* Make sure you use Python 2.

Installation
----------------------

### Create the environment and install the requirements
 
* Clone this repo to your computer by typing `git clone https://github.com/vertcli/machine_learning.git`.
* Get into the folder using `cd machine_learning/glass_classification`.
* Create a python environment using `conda create --name glass_classification --file requirements.txt python=2`.
    
### Download the data

* Activate this environment: `source activate glass_classification`.
* Run `mkdir data`.
* Run `python glass_data_reader.py`.
    * This will create `raw_glass.csv` in the `data` folder. The data will look like:
    ![alt tag](https://github.com/vertcli/machine_learning/blob/master/glass_classification/img/raw_data.jpg)

Usage
-----------------------

* If you are not already in glass_classification folder get into it. Also, make sure you have glass_classification environment active.
* Run `mkdir processed` to create a directory for our processed datasets.
* Run `python glass_process_data.py` normalize and center the dataset.
    * This will create `data.csv` in the `processed` folder. The data will look like:
    ![alt tag](https://github.com/vertcli/machine_learning/blob/master/glass_classification/img/processed_data.jpg)    
* Run `python glass_pca.py`.
    * This will show up the variance of each feature in the new space. Notice that the first eigenvalue is the one with more variance, meaning that our data lives mainly in a line.
    * It will also export this plot into the `img` folder.
* OPTIONAL: Run `python glass_visualization.py`.
    * This will show up a pair-plot to see the correlation between the features of our dataset.
    * It will also export this plot into the `img` folder.
* Run `python glass_ada_boost_classifier.py`.
    * This will run one-vs-all adaptive boosting classifier with different number of estimators. The dataset is divided into 10 folks and we do a cross validation across the folks. The output of the algorithm is one accuracy score for each training and a graph with box-plot of the accuracy vs the number of estimators. Note: this graph is also exported to the `img` folder.

Extending this
-------------------------

If you want to extend this work, here are a few places to start:

* You can explore how the algorithm behave with different number of estimators changing the file `settings.py`.
* Change the base estimator ADA BOOST use.