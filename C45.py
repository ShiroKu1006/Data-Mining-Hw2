import pandas as pd
import numpy as np
from chefboost import Chefboost as chef
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("data\\processed_bank_data_final.csv")

target = "TransactionType"

df[target] = df[target].astype(str)
feature_cols = [c for c in df.columns if c not in [target, "dataset_group"]]

kf = KFold(n_splits=10, shuffle=True, random_state=42)

acc_list, precision_list, recall_list, f1_list, auc_list = [], [], [], [], []

for train_idx, test_idx in kf.split(df):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    model = chef.fit(
        train_df[feature_cols + [target]],
        config={"algorithm": "C4.5"},
        target_label=target
    )


    X_test = test_df[feature_cols].values.tolist()
    y_pred = np.array([chef.predict(model, x) for x in X_test])
    y_true = test_df[target].values

    acc_list.append(accuracy_score(y_true, y_pred))
    precision_list.append(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_list.append(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_list.append(f1_score(y_true, y_pred, average="weighted", zero_division=0))


    y_true_bin = (y_true == "Credit").astype(int)
    y_pred_bin = (y_pred == "Credit").astype(int)
    auc_list.append(roc_auc_score(y_true_bin, y_pred_bin))

print("Model: C4.5")
print(f"Accuracy : {np.mean(acc_list):.4f}")
print(f"Precision: {np.mean(precision_list):.4f}")
print(f"Recall   : {np.mean(recall_list):.4f}")
print(f"F1-score : {np.mean(f1_list):.4f}")
print(f"AUC      : {np.mean(auc_list):.4f}")
