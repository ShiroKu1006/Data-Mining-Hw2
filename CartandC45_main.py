from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import pandas as pd

def evaluate_model_cv(model, X, y, model_name="Model"):
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring_metrics)
    results = {
        'Algorithm': model_name,
        'Accuracy': cv_results['test_accuracy'].mean(),
        'Precision': cv_results['test_precision'].mean(),
        'Recall': cv_results['test_recall'].mean(),
        'F-Measure': cv_results['test_f1'].mean(),
        'AUC': cv_results['test_roc_auc'].mean()
    }
    return results

def show_result(results):
    print(f"Model    : {results['Algorithm']}")
    print(f"Ac       : {results['Accuracy']:.4f}")
    print(f"Recall   : {results['Recall']:.4f}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"F-Measure: {results['F-Measure']:.4f}")
    print(f"AUC      : {results['AUC']:.4f}")
    print()

df = pd.read_csv("data\\processed_bank_data_final.csv")

if 'dataset_group' in df.columns:
    df = df.drop(columns=['dataset_group'])

target = 'isFraud'
X = df.drop(columns=[target])
y = df[target]

cart_model = DecisionTreeClassifier(random_state=42)
results = evaluate_model_cv(cart_model, X, y, "CART (sklearn)")
show_result(results)

c45_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
results_c = evaluate_model_cv(c45_model, X, y, "C4.5 (sklearn entropy)")
show_result(results_c)
