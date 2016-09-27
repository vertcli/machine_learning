
# coding: utf-8

# In[1]:

# Import the required libraries:
import pandas as pd

# Read the raw data and then normalize and center it:
pd_glass = pd.read_csv('data/raw_data.csv')

# We visualize a pair plot with all features vs all features so we can see if it is possible to split between classes
pd_glass = (pd_glass - pd_glass.mean()) / (pd_glass.max() - pd_glass.min())

pd_glass.to_csv('processed/data.csv')

