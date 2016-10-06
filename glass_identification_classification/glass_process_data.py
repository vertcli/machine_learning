
#######################################################
# Author: Albert Climent Bigas (vert.cli@gmail.com)
# Description: Script preprocess the dataset.
#######################################################

# Import the required libraries:
import pandas as pd
import settings as st

print('Processing data...')

# Read the raw data and then normalize and center it:
pd_glass = pd.read_csv('data/raw_data.csv')

# We visualize a pair plot with all features vs all features so we can see if it is possible to split between classes
feature_names = st.FEATURE_NAMES
pd_glass[feature_names] = (pd_glass[feature_names] - pd_glass[feature_names].mean()) / (pd_glass[feature_names].max() - pd_glass[feature_names].min())

pd_glass.to_csv('data/processed_data.csv', index = False)

print('Done. Processed dataset head:')
print(pd_glass.head(5))
