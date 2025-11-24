import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("data\\processed_bank_data_final.csv")

target = "TransactionType"

feature_cols = [c for c in df.columns if c not in 
               [target, "dataset_group",
                "TransactionType_Credit", "TransactionType_Debit"]]

X = df[feature_cols]
y = df[target].astype(str)  

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

acc_list, precision_list, recall_list, f1_list, auc_list = [], [], [], [], []

for train_idx, test_idx in kf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall_list.append(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_list.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    y_test_bin = (y_test == "Credit").astype(int)
    y_pred_bin = (y_pred == "Credit").astype(int)
    auc_list.append(roc_auc_score(y_test_bin, y_pred_bin))

print("Model: CART")
print(f"Accuracy : {np.mean(acc_list):.4f}")
print(f"Precision: {np.mean(precision_list):.4f}")
print(f"Recall   : {np.mean(recall_list):.4f}")
print(f"F1-score : {np.mean(f1_list):.4f}")
print(f"AUC      : {np.mean(auc_list):.4f}")
