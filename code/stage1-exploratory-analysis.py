#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("../data/student_por_mod.csv")

print(df)

df.describe(percentiles=[.05, .10, .25, .50, .75, .90, .95])

print(df.corr())

fig = plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='plasma')

# TODO: write loop that generates pngs of histograms/boxplots/... of each pair of variables
