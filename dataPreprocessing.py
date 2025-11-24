import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/bank_transactions_data_2.csv")

dates = ["TransactionDate", "PreviousTransactionDate"]
for c in dates:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

if set(dates).issubset(df.columns):
    df['DaysSincePrev'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.days
    df['TxnHour'] = df['TransactionDate'].dt.hour
    df['TxnDayOfWeek'] = df['TransactionDate'].dt.dayofweek
    df['DaysSincePrev'] = df['DaysSincePrev'].fillna(-1)

cols_to_drop = [
    'TransactionID', 'AccountID', 'DeviceID', 'IP Address', 'MerchantID',
    'TransactionDate', 'PreviousTransactionDate'
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

target_col = 'TransactionType'

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    verbose_feature_names_out=False
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

X_train_final = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_final = pd.DataFrame(X_test_processed, columns=feature_names)

df_train = X_train_final.copy()
df_train[target_col] = y_train.values
df_train['dataset_group'] = 'train'

df_test = X_test_final.copy()
df_test[target_col] = y_test.values
df_test['dataset_group'] = 'test'

df_t = pd.concat([df_train, df_test], axis=0, ignore_index=True)
out = "data\\processed_bank_data_final.csv"
df_t.to_csv(out, index=False)
