Glass Identification Classification
-----------------------------------

The aim of this project is to deal with a multiclass classification problem using glass information. After training the algorithm we will predict which type of glass each sample have. The dataset used for this problem is public and can be found [here](https://archive.ics.uci.edu/ml/datasets/Glass+Identification) and it contains 214 samples and 9 features. As you can read in this link, the study of glass classification was motivated by criminological investigations (i.e. determining the type of glass left on a scene of a crime can be used as evidence).


Installation
----------------------

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.
    * Make sure you use Python 2.
    * You may want to use a virtual environment for this.
    
### Download the data

* Clone this repo to your computer.
* Get into the folder using `cd glass_classification`.
* Run `mkdir data`.
* Run `python glass_data_reader.py`.
    * This will create `raw_glass.csv` in the `data` folder.

Usage
-----------------------

* Run `mkdir processed` to create a directory for our processed datasets.
* Run `python glass_process_data.py` normalize and center the dataset.
    * This will create `data.csv` in the `processed` folder.
* Run `mkdir img` to create a directory to save the plots as images.    
* Run `python glass_pca.py`.
    * This will show up the variance of each feature in the new space. Notice that the first and second axis are the ones with more variance.
    * It will also export this plot into the `img` folder.
* Run `python glass_visualization.py`.
    * This will show up a pair-plot to see the correlation between the features of our dataset.
    * It will also export this plot into the `img` folder.
* Run `python glass_ada_boost_classifier.py`.
    * This will run one-vs-all adaptive boosting classifier with different number of classifiers. The dataset is divided into X folds and we do a cross validation across the folds. The output of the algorithm is one accuracy score for each training and a graph with box-plot of the accuracy vs the number of classifiers. Note: this graph is also exported to the `img` folder.

TODO
Extending this
-------------------------

If you want to extend this work, here are a few places to start:

* Generate more features in `annotate.py`.
* Switch algorithms in `predict.py`.
* Add in a way to make predictions on future data.
* Try seeing if you can predict if a bank should have issued the loan.
    * Remove any columns from `train` that the bank wouldn't have known at the time of issuing the loan.
        * Some columns are known when Fannie Mae bought the loan, but not before
    * Make predictions.
* Explore seeing if you can predict columns other than `foreclosure_status`.
    * Can you predict how much the property will be worth at sale time?
* Explore the nuances between performance updates.
    * Can you predict how many times the borrower will be late on payments?
    * Can you map out the typical loan lifecycle?