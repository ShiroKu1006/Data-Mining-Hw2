import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from CHAID import Tree

df = pd.read_csv("data\\processed_bank_data_final.csv")

if 'dataset_group' in df.columns:
    df = df.drop(columns=['dataset_group'])

target = 'isFraud'
X = df.drop(columns=[target])
y = df[target]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

num_high_card = [c for c in num_cols if X[c].nunique() > 10]
num_low_card = [c for c in num_cols if X[c].nunique() <= 10]

if num_high_card:
    kbd = KBinsDiscretizer(
        n_bins=3,
        encode='ordinal',
        strategy='quantile',
        quantile_method='averaged_inverted_cdf'
    )
    X_high_binned = pd.DataFrame(
        kbd.fit_transform(X[num_high_card]),
        columns=num_high_card,
        index=X.index
    )
    for c in num_high_card:
        X_high_binned[c] = X_high_binned[c].astype('category')
else:
    X_high_binned = pd.DataFrame(index=X.index)

X_low = X[num_low_card].copy()
for c in num_low_card:
    X_low[c] = X_low[c].astype('category')

X_cat = X[cat_cols].copy()
for c in cat_cols:
    X_cat[c] = X_cat[c].astype('category')

X_final = pd.concat([X_high_binned, X_low, X_cat], axis=1)
df_final = pd.concat([X_final, y], axis=1)

feature_cols = [c for c in df_final.columns if c != target]
variable_types = {col: 'nominal' for col in feature_cols}

df_final[target] = df_final[target].astype('category')

tree = Tree.from_pandas_df(
    df_final[feature_cols + [target]],
    variable_types,
    target
)

tree.print_tree()
tree.render(path="chaid_tree", view=False)
