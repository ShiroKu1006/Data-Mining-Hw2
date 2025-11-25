import pandas as pd
import numpy as np
from CHAID import Tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv("data\\processed_bank_data_final.csv")

target = "TransactionType"
feature_cols = [
    c for c in df.columns
    if c not in [
        target,
        "dataset_group",
        "TransactionType_Credit",
        "TransactionType_Debit"
    ]
]

numeric_cols = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "DaysSincePrev"
]

discretizer = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
df[numeric_cols] = discretizer.fit_transform(df[numeric_cols])

df[target] = df[target].astype(str)

i_variables = {col: "nominal" for col in feature_cols}

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

acc_list, precision_list, recall_list, f1_list, auc_list = [], [], [], [], []


def predict_with_chaid(tree, X):
    rules_list = tree.classification_rules()
    root_node = tree.tree_store[0]
    default_label = max(root_node.members, key=root_node.members.get)

    if not rules_list:
        return np.array([default_label] * len(X))

    node_major_label = {}
    for r in rules_list:
        node = tree.tree_store[r["node"]]
        node_major_label[r["node"]] = max(node.members, key=node.members.get)

    preds = []
    for _, row in X.iterrows():
        assigned = False
        for r in rules_list:
            ok = all(row[c["variable"]] in c["data"] for c in r["rules"])
            if ok:
                preds.append(node_major_label[r["node"]])
                assigned = True
                break
        if not assigned:
            preds.append(default_label)
    return np.array(preds)


def predict_proba_with_chaid(tree, X, positive_label="Credit"):
    rules_list = tree.classification_rules()
    root_node = tree.tree_store[0]

    root_total = sum(root_node.members.values())
    if root_total > 0:
        default_prob = root_node.members.get(positive_label, 0) / root_total
    else:
        default_prob = 0.0

    if not rules_list:
        return np.array([default_prob] * len(X))

    node_pos_prob = {}
    for r in rules_list:
        node = tree.tree_store[r["node"]]
        total = sum(node.members.values())
        if total > 0:
            node_pos_prob[r["node"]] = node.members.get(positive_label, 0) / total
        else:
            node_pos_prob[r["node"]] = default_prob

    scores = []
    for _, row in X.iterrows():
        prob = default_prob
        for r in rules_list:
            if all(row[c["variable"]] in c["data"] for c in r["rules"]):
                prob = node_pos_prob[r["node"]]
                break
        scores.append(prob)

    return np.array(scores)


for train_idx, test_idx in kf.split(df[feature_cols], df[target]):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    tree = Tree.from_pandas_df(
        train_df[feature_cols + [target]],
        i_variables,
        target,
        max_depth=5,
        min_child_node_size=50
    )

    X_test = test_df[feature_cols]
    y_true = test_df[target].values

    y_pred = predict_with_chaid(tree, X_test)
    y_score = predict_proba_with_chaid(tree, X_test, positive_label="Credit")

    acc_list.append(accuracy_score(y_true, y_pred))
    precision_list.append(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_list.append(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_list.append(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    y_true_bin = (y_true == "Credit").astype(int)
    auc_list.append(roc_auc_score(y_true_bin, y_score))

print("Model : CHAID")
print(f"Accuracy : {np.mean(acc_list):.4f}")
print(f"Precision: {np.mean(precision_list):.4f}")
print(f"Recall   : {np.mean(recall_list):.4f}")
print(f"F1-score : {np.mean(f1_list):.4f}")
print(f"AUC      : {np.mean(auc_list):.4f}")
