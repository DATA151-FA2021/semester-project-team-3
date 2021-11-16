import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic

df = pd.read_csv("../data/student_por_mod.csv")

df["failures"] = df["failures"].map({0: "no_fail", 1: "fail", 2: "fail", 3: "fail"})

df["failures"] = np.where(df.failures == "fail", 1, 0)


# Obtained through feature importance graph, the variables that gave the highest
# score.
predictors = ["G3", "age", "G1", "absences", "Medu", "goout", "Fedu"]


x = pd.get_dummies(df[predictors], drop_first=True)
print(x.head())

y = df["failures"]

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=123, stratify=y
)


# First Model: Decision Tree

d_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
d_tree.fit(x_train, y_train)
y_pred = d_tree.predict(x_val)
print(accuracy_score(y_val, y_pred))

y_pred = d_tree.predict_proba(x_val)[:, 1]

labels = np.unique(y_val)
cm = confusion_matrix(y_val, y_pred > 0.2, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

for func in [recall_score, precision_score, f1_score]:
    print(f"{func} :  {func(y_val, y_pred > 0.2, average = 'weighted')}")

# Second Model: Random Forest

r_forest = RandomForestClassifier(n_estimators=500, random_state=7)
r_forest.fit(x_train, y_train)
y_pred = r_forest.predict_proba(x_val)[:, 1]

labels = np.unique(y_val)
cm = confusion_matrix(y_val, y_pred > 0.2, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

for func in [accuracy_score, recall_score, precision_score, f1_score]:
    print(f"{func} : {func(y_val, y_pred > 0.2, average = 'weighted')}")


# Code above runs, have not yet refactored the rest of the doc.

"""#SMOTE"""

sm = SMOTE(sampling_strategy="minority", random_state=2)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)

RF = RandomForestClassifier(random_state=2)
RF.fit(x_train_res, y_train_res)
predictions = RF.predict(x_val)

# print classification report
print(metrics.classification_report(y_val, predictions))

# variable importance
importances = r_forest.feature_importances_
# Plotting variable importance
plt.figure(figsize=(5, 10))
sorted_idx = r_forest.feature_importances_.argsort()
plt.barh(x.columns[0:][sorted_idx], r_forest.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.title("Random Forest Feature Imortance by Group 3")
plt.savefig("figure3.png", dpi=150, bbox_inches="tight")

# importing roc metrics from sklearn

# Extracting probabilities
rf_probs = pd.Series(RF.predict_proba(x_val)[:, 1])

# calculate scores
rf_auc = roc_auc_score(y_val, rf_probs)

# summarize scores
print(f"Random Forest: ROC AUC = {rf_auc:.3f}")

# calculate roc curves
rf_fpr, rf_tpr, _ = roc_curve(y_val, rf_probs)

# plot the roc curve for the model
# plt.plot(rf_fpr, rf_tpr, marker='.', label='Logistic')
plt.plot(rf_fpr, rf_tpr, color="darkorange", lw=3)
# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("figure5.png", dpi=150, bbox_inches="tight")

"""#Logistic Regression"""

df = df.drop(
    [
        "G1",
        "G2",
        "G3",
        "absences",
        "age",
        "Medu",
        "Fedu",
        "traveltime",
        "studytime",
        "famrel",
        "freetime",
        "goout",
        "Walc",
        "Dalc",
        "health",
    ],
    axis=1,
)

# Using train_test_split function to split the dataset
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.3, random_state=1234, stratify=y
)

# Chi Square test of Independence
for col in df.columns[1:]:
    crosstab = pd.crosstab(df[col], df["failures"], margins=True)
    stat, p, dof, expected = chi2_contingency(crosstab)
    print("P value of Chi Square between failures and", col, "is", p)

mosaic(data=df, index=["reason", "failures"])
plt.title("Mosaic Plot of Reason and Failures by Group 3")

df["failures"].value_counts()

"""#Model 3: Logistic Regression"""

# Choose Predictors from the dataset
predictors = df.columns[[7, 8, 15]]

# Specify the target variable
target = "failures"

# Target variable
y = df[target]
y[0:5]

x = pd.get_dummies(df[predictors], drop_first=True)
x.head()

# Importing logistic regression
lr_clf = LogisticRegression()
lr_clf.fit(x_train, y_train)

# Prediction using the logistic regression
y_pred = lr_clf.predict_proba(x_val)[:, 1]
y_pred[0:10]  # first coloumn is prob of negative class (fail)

# Confusion matrix
labels = np.unique(y_val)
cm = confusion_matrix(y_val, y_pred > 0.2, labels=labels)
pd.DataFrame(
    cm, index=labels, columns=labels
)  # lower the threshold, increase the sensitivity 20% Chance of fail

y.value_counts()

y_train.value_counts()

# Evaluating performance
print("Accuracy score is:", accuracy_score(y_val, y_pred > 0.2))
print("Recall score is:", recall_score(y_val, y_pred > 0.2))
print("Precision score is:", precision_score(y_val, y_pred > 0.2))
print("F1 score is:", f1_score(y_val, y_pred > 0.2))

# Cross-validation
scores = cross_val_score(d_tree, x_train, y_train, cv=10, scoring="recall")
print("The cross validation scores are", scores)
print("The mean score is", scores.mean())

# Cross-validation splitter as a cv parameter
shuffle_split = StratifiedShuffleSplit(test_size=0.2, n_splits=10, random_state=123)
scores = cross_val_score(d_tree, x_train, y_train, cv=shuffle_split, scoring="recall")
print("The cross validation scores are", scores)
print("The mean score is", scores.mean())

# Extracting probabilities
lr_probs = pd.Series(lr_clf.predict_proba(x_val)[:, 1])


# calculate scores
lr_auc = roc_auc_score(y_val, lr_probs)

# summarize scores
print(f"Logistic: ROC AUC = {lr_auc:.3f}")

# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_val, lr_probs)

# plot the roc curve for the model
# plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.plot(lr_fpr, lr_tpr, color="darkorange", lw=3)
# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
