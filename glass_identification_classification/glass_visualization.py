#######################################################
# Author: Albert Climent Bigas (vert.cli@gmail.com)
# Description: Script to visualize all the dataset
# on a one vs one variable plot.
#######################################################

# Import the required libraries:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read processed data:
pd_glass = pd.read_csv('data/processed_data.csv')

# We visualize a pair plot with all features vs all features so we can see if it is possible to split between classes
sns.pairplot(pd_glass,hue='Type')

plt.title('Data visualization')
figure = plt.gcf()
plt.show()

figure.savefig("img/pair_plot.png")
print('Done!')

