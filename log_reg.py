import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# Load dataset
test_id = pd.read_csv("test_id_small.csv", index_col=[0])
test_tr = pd.read_csv("test_tr_small.csv", index_col=[0])
train_id = pd.read_csv("train_id_small.csv", index_col=[0])
train_tr = pd.read_csv("train_tr_small.csv", index_col=[0])

# Merge data
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
del test_id, test_tr, train_id, train_tr

# 
submission = pd.DataFrame({'TransactionID':test.TransactionID})

# Missing values
combined = pd.concat([train.drop(columns=['isFraud', 'TransactionID']), test.drop(columns='TransactionID')])
len_train = len(train)

y = train['isFraud']

# Dropping columns with more than 25% missing values 
misval = combined.isnull().sum()/len(combined)
combined_mv = combined.drop(columns=misval[misval>0.25].index)
del combined

# Filling missing values
num = combined_mv.select_dtypes(include=np.number) 
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
num_df = pd.DataFrame(imp.fit_transform(num), columns=num.columns)
del imp, num

cat = combined_mv.select_dtypes(exclude=np.number)
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
cat_df = pd.DataFrame(imp.fit_transform(cat), columns=cat.columns)
del imp, cat

# Combine
combined_clean = pd.concat([num_df, cat_df], axis=1)
del num_df, cat_df, combined_mv

# One-hot encoding
combined_encoded = pd.get_dummies(combined_clean, drop_first=True)
del combined_clean

# Separate test & train splitting ved 80%
X = combined_encoded.iloc[:len_train]
test = combined_encoded.iloc[len_train:]
train = pd.concat([X,y], axis = 1)
del combined_encoded, X

train.sort_values('TransactionDT', inplace=True)

X = train.drop(['isFraud'], axis=1)
y = train['isFraud']

X_train = X.iloc[:int(len(X)*.8)]
y_train = y.iloc[:int(len(X)*.8)]
X_val = X.iloc[int(len(X)*.8):]
y_val = y.iloc[int(len(X)*.8):]

# Logistic regression
logreg = LogisticRegression(max_iter=10000, solver = "sag", random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict_proba(X_val)

auc = roc_auc_score(y_val, y_pred[:, 1])
print("Validation AUC =",auc)

### Prediction submission
pred = logreg.predict_proba(test)
submission['isFraud'] = pred[:, 1]

print(submission.head())
