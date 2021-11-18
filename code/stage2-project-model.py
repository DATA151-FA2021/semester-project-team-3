import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random as rand
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

rand.seed(10)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y)


def runModel(model, xt, yt, xv, yv):
    model.fit(xt, yt)
    y_pred = model.predict(xv)
    y_pred_prob = model.predict_proba(xv)[:, 1]
    print(y_pred_prob[0:10])  # first coloumn is prob of negative class (fail)
    labels = np.unique(yv)
    cm = confusion_matrix(yv, y_pred_prob > 0.2, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    print(f"{accuracy_score.__name__} : {accuracy_score(yv, y_pred_prob > 0.2)}")
    for func in [recall_score, precision_score, f1_score]:
        print(f"{func.__name__} :  {func(yv, y_pred_prob > 0.2, average = 'weighted')}")

    # print classification report
    print(metrics.classification_report(yv, y_pred))
    # calculate scores & Extracting probabilities
    auc = roc_auc_score(yv, pd.Series(model.predict_proba(xv)[:, 1]))
    # summarize scores
    print(f"ROC AUC : {auc:.3f}")

    plt.figure()
    m_fpr, m_tpr, _ = roc_curve(yv, pd.Series(y_pred_prob))
    plt.plot(m_fpr, m_tpr, color="darkorange", lw=3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(f"{model}.png", dpi=150, bbox_inches="tight")

    # Cross-validation
    scores = cross_val_score(model, xt, yt, cv=10, scoring="recall")
    print(f"cross validation : {scores}\nmean : {scores.mean()}")

    # Cross-validation splitter as a cv parameter
    shuffle_split = StratifiedShuffleSplit(
        test_size=0.2,
        n_splits=10,
        random_state=123,
    )
    scores = cross_val_score(model, xt, yt, cv=shuffle_split, scoring="recall")
    print(f"cross validation : {scores}\nmean : {scores.mean()}")

# First Model: Decision Tree

d_tree = DecisionTreeClassifier(max_depth=3, random_state=0)
runModel(d_tree, x_train, y_train, x_val, y_val)

# Second Model: Random Forest

r_forest = RandomForestClassifier(n_estimators=500, random_state=7)
runModel(r_forest, x_train, y_train, x_val, y_val)

# Plotting variable importance for Random Forest
plt.figure(figsize=(5, 10))
sorted_idx = r_forest.feature_importances_.argsort()
plt.barh(x.columns[0:][sorted_idx], r_forest.feature_importances_[sorted_idx])
plt.xlabel(f"{r_forest} Feature Importance")
plt.title(f"{r_forest} Feature Imortance by Group 3")
plt.savefig(f"{r_forest}_feature_importance.png", dpi=150, bbox_inches="tight")

# Third Model: Random Forest with SMOTE undersampling

sm = SMOTE(sampling_strategy="minority", random_state=2)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)

RF = RandomForestClassifier(random_state=2)
runModel(RF, x_train_res, y_train_res, x_val, y_val)

# Fourth Model: Logistic Regression

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
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, stratify=y, random_state=1234)

# Chi Square test of Independence
for col in df.columns[1:]:
    crosstab = pd.crosstab(df[col], df["failures"], margins=True)
    stat, p, dof, expected = chi2_contingency(crosstab)
    print("P value of Chi Square between failures and", col, "is", p)

mosaic(data=df, index=["reason", "failures"])
plt.title("Mosaic Plot of Reason and Failures by Group 3")

print(df["failures"].value_counts())

predictors = df.columns[[7, 8, 15]]  # Choose Predictors from the dataset

x = pd.get_dummies(df[predictors], drop_first=True)
x.head()

lr_clf = LogisticRegression()
runModel(lr_clf, x_train, y_train, x_val, y_val)
