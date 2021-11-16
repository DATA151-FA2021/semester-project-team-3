import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

df = pd.read_csv("../data/student_por_mod.csv")

df['failures' = df['failures'].map({0: 'no_fail', 1: 'fail', 2: 'fail', 3: 'fail'})

df['failures'] = np.where(df.failures == 'fail', 1, 0)


# Obtained through feature importance graph, the variables that gave the highest
# score.
predictors = ['G3', 'age', 'G1', 'absences', 'Medu', 'goout', 'Fedu']
 
   
x = pd.get_dummies(df[predictors], drop_first = True)
print(x.head())
   
y = df['failures']

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=123, stratify = y)   

   
# First Model: Decision Tree
   
d_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
d_tree.fit(x_train, y_train)
y_pred = d_tree.predict(x_val)
print(accuracy_score(y_val, y_pred))

y_pred = d_tree.predict_proba(x_val)[:,1]

labels = np.unique(y_val)
cm = confusion_matrix(y_val, y_pred > 0.2, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))
   
for func in [recall_score, precision_score, f1_score]:
   print(f"{func} :  {func(y_val, y_pred > 0.2, average = 'weighted')}")

