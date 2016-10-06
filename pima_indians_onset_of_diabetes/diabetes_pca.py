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
pd_diabetes = pd.read_csv('data/processed_data.csv')

#print(pd_diabetes.head(5))
x = pd_diabetes.as_matrix(st.FEATURE_NAMES)
y = pd_diabetes.as_matrix(['Class'])

# Perform PCA over our data:
print('Performing PCA...')
pca = decomposition.PCA()
pca.fit(x)

# Plot the covariances of our samples over the new frame:
print('Generating plots...')
sns.barplot(range(8), np.sort(np.diagonal(pca.get_covariance()))[::-1], palette="BuGn_d")

# Save the figure:
plt.xlabel('$\lambda_i$')
plt.ylabel('Variance')
plt.title('PCA')
figure = plt.gcf()
plt.show()

figure.savefig("img/pca_variances.png")

# Show the dataset on the new space.
pca = decomposition.PCA(n_components=2)
pca.fit(x)
p = pca.fit_transform(x)
p = pd.DataFrame(data=np.concatenate((p,y),axis=1),columns=['x','y','Class'])
sns.pairplot(p,hue='Class')

# Save the figure:
plt.title('2D data projection')
figure2 = plt.gcf()
plt.show()

figure2.savefig("img/pca_data_projection.png")
print('Done!')




