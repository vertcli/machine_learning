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
import pandas as pd
import settings as st

print('Reading data...')

# Read the processed data:
pd_glass = pd.read_csv('data/processed_data.csv')

#print(pd_glass.head(5))
x = pd_glass.as_matrix(st.FEATURE_NAMES)
y = pd_glass.as_matrix(['Type'])

# Perform PCA over our data:
print('Performing PCA...')
pca = decomposition.PCA()
pca.fit(x)

# Plot the covariances of our samples over the new frame:
print('Generating plot...')
sns.barplot(range(9), np.sort(np.diagonal(pca.get_covariance()))[::-1], palette="BuGn_d")

# Save the figure:
plt.xlabel('$\lambda_i$')
plt.ylabel('Variance')
plt.title('PCA')
figure = plt.gcf()
plt.show()

figure.savefig("img/pca_variances.png")
print('Done!')




