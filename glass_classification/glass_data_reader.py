#######################################################
# Author: Albert Climent Bigas (vert.cli@gmail.com)
# Description: Script to download Glass Identification
# Classification dataset from url.
#######################################################

# Import the required libraries:
import numpy as np
import pandas as pd
import urllib as url

# We define a function to convert ndarrays to pandas DataFrames:
def dataset_to_dataframe(dataset, feature_names):
    df = pd.DataFrame(dataset, columns = feature_names)
    return df

print('Downloading data...')

# Load Glass Identification Dataset from UCI:
raw_data = url.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data')
glass = np.loadtxt(raw_data, delimiter=',')
glass = np.delete(glass,0,1)

# In order to plot and visualize the data, we change to panda's DataFrame structure:
pd_glass = dataset_to_dataframe(glass, feature_names=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type'])

# Save the data to our system:
pd_glass.to_csv('data/raw_data.csv')

print('Done!')
