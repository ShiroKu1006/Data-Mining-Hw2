import pandas as pd
import numpy as np
from chefboost import Chefboost as chef
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("data\\processed_bank_data_final.csv")

target = "TransactionType"
df[target] = df[target].astype(str)

feature_cols = [
    c for c in df.columns
    if c not in [
        target,
        "dataset_group",
        "TransactionType_Credit",
        "TransactionType_Debit"
    ]
]

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

acc_list, precision_list, recall_list, f1_list, auc_list = [], [], [], [], []

for train_idx, test_idx in kf.split(df[feature_cols], df[target]):
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    counts = train_df[target].value_counts()
    max_count = counts.max()

    balanced_parts = []
    for cls, cnt in counts.items():
        sub = train_df[train_df[target] == cls]
        if cnt < max_count:
            sub_over = sub.sample(max_count - cnt, replace=True, random_state=42)
            sub = pd.concat([sub, sub_over], axis=0)
        balanced_parts.append(sub)

    train_balanced = pd.concat(balanced_parts, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    model = chef.fit(
        train_balanced[feature_cols + [target]],
        config={"algorithm": "C4.5"},
        target_label=target
    )

    X_test = test_df[feature_cols].values.tolist()
    y_true = test_df[target].values
    y_pred = np.array([chef.predict(model, x) for x in X_test])

    acc_list.append(accuracy_score(y_true, y_pred))
    precision_list.append(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_list.append(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_list.append(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    y_true_bin = (y_true == "Credit").astype(int)
    y_pred_bin = (y_pred == "Credit").astype(int)
    auc_list.append(roc_auc_score(y_true_bin, y_pred_bin))

print("Model: C4.5 ")
print(f"Accuracy : {np.mean(acc_list):.4f}")
print(f"Precision: {np.mean(precision_list):.4f}")
print(f"Recall   : {np.mean(recall_list):.4f}")
print(f"F1-score : {np.mean(f1_list):.4f}")
print(f"AUC      : {np.mean(auc_list):.4f}")

print(df[target].value_counts(normalize=True))
from collections import Counter
print("y_true:", Counter(y_true))
print("y_pred:", Counter(y_pred))
