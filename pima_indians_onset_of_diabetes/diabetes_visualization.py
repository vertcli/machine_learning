#######################################################
# Author: Albert Climent Bigas (vert.cli@gmail.com)
# Description: Python script to visualize PIMA Indians
# Diabetes dataset on a pair-plot.
######################################################

# Import the required libraries:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import settings as st

print('Reading data...')

# Read the raw data and then normalize and center it:
pd_diabetes = pd.read_csv('data/raw_data.csv')

print('Generating plot...')
# We visualize a pair plot with all features vs all features so we can see if it is possible to split between class 0 and 1:
sns.pairplot(pd_diabetes,hue='Class')

plt.title('Data visualization')
figure = plt.gcf()
plt.show()

figure.savefig("img/pair_plot.png")
print('Done!')

