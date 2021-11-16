#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("../data/student_por_mod.csv")

# FIXME: convert string values in dataset to be numeric - it will allow us to do the comparasions later

df = df.replace(to_replace="no", value=0)
df = df.replace(to_replace="yes", value=1)

df = df.replace(to_replace="F", value=0)
df = df.replace(to_replace="M", value=1)

df = df.replace(to_replace="GP", value=0)
df = df.replace(to_replace="MS", value=1)

df = df.replace(to_replace="A", value=0)
df = df.replace(to_replace="T", value=1)

print(df)

df.describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

print(df.corr())

fig = plt.figure(figsize=(15, 15))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap="plasma")
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

        # Recreate the fig - hopefully prevents the two plots from being merged.
        fig = plt.figure(figsize=(8, 8))

        # Generate Bi-Variate Histogram
        hp = sns.histplot(data=df, x=i_column, y=j_column)
        fig.suptitle(f"{i_column} vs {j_column}")
        plt.savefig(f"{i_column}_vs_{j_column}_histplot.png")
