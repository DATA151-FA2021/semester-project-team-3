import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/student_por_mod.csv")

df['failures' = df['failures'].map({0: 'no_fail', 1: 'fail', 2: 'fail', 3: 'fail'})

df['failures'] = np.where(df.failures == 'fail', 1, 0)


# Obtained through feature importance graph, the variables that gave the highest
# score.
predictors = ['G3', 'age', 'G1', 'absences', 'Medu', 'goout', 'Fedu']
 
   
x = pd.get_dummies(df[predictors], drop_first = True)
print(x.head())
   
y = df['failures']
