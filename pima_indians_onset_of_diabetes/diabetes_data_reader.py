#######################################################
# Author: Albert Climent Bigas (vert.cli@gmail.com)
# Description: Script to download PIMA Indians onset
# of Diabetes dataset from url.
#######################################################

# Import the required libraries:
import numpy as np
import pandas as pd
import urllib as url
import settings as st

# We define a function to convert ndarrays to pandas DataFrames:
def dataset_to_dataframe(dataset, feature_names):
    df = pd.DataFrame(dataset, columns = feature_names)
    return df

print('Downloading data...')

# Load Pima Indians Diabetes Dataset from UCI:
raw_data = url.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data')
diabetes = np.loadtxt(raw_data, delimiter=',')

# In order to plot and visualize the data, we change to panda's DataFrame structure:
pd_diabetes = dataset_to_dataframe(diabetes, feature_names=st.DATASET_NAMES)

# Save the data to our system:
pd_diabetes.to_csv('data/raw_data.csv', index = False)

print('Done. Dataset head:')
print(pd_diabetes.head(5))
