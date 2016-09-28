#######################################################
# Author: Albert Climent Bigas (vert.cli@gmail.com)
# Description: PCA analysis over Glass Identification
# dataset.
######################################################

# Import the required libraries:
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the processed data:
glass = np.genfromtxt('processed/data.csv', delimiter=',')

x = np.delete(glass,9,1)
y = glass[:,9]

# Perform PCA over our data:
pca = decomposition.PCA()
pca.fit(x)

# Plot the covariances of our samples over the new frame:
sns.barplot(range(9), np.sort(np.diagonal(pca.get_covariance()))[::-1], palette="BuGn_d")
plt.show()




