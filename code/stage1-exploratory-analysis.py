#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("../data/student_por_mod.csv")

# FIXME: convert string values in dataset to be numeric - it will allow us to do the comparasions later

print(df)

df.describe(percentiles=[.05, .10, .25, .50, .75, .90, .95])

print(df.corr())

fig = plt.figure(figsize=(15,15))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='plasma')
fig.suptitle("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", dpi=200)

for i_column in df.columns:
  for j_column in df.columns.drop([i_column]):
    # Create a 8x8 Figure
    fig = plt.figure(figsize=(8, 8))
    
    # Generate BoxPlot
    bp = sns.boxplot(data=df, x=i_column, y=j_column)
    fig.suptitle(f"{i_column} vs {j_column}")
    plt.savefig(f"{i_column}_vs_{j_column}_boxplot.png")
    
    # Generate Bi-Variate Histogram
    hp = sns.histplot(data=df, x=i_column, y=j_column)
    fig.suptitle(f"{i_column} vs {j_column}")
    plt.savefig(f"{i_column}_vs_{j_column}_histplot.png")

