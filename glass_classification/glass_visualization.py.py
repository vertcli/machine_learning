# Import the required libraries:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read processed data:
pd_glass = pd.read_csv('processed/data.csv')

# We visualize a pair plot with all features vs all features so we can see if it is possible to split between classes
sns.pairplot(pd_glass,hue='Type')
sns.plt.show()

