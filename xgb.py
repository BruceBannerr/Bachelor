from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.metrics import log_loss

from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve

# Load dataset
test_id = pd.read_csv("test_identity.csv")
test_tr = pd.read_csv("test_transaction.csv")
train_id = pd.read_csv("train_identity.csv")
train_tr = pd.read_csv("train_transaction.csv")

# Merge data
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
del test_id, test_tr, train_id, train_tr

# Submission
submission = pd.DataFrame({'TransactionID': test.TransactionID})

# Missing values
combined = pd.concat([train.drop(
    columns=['isFraud', 'TransactionID']), test.drop(columns='TransactionID')])
len_train = len(train)

y = train['isFraud']

# Dropping columns with more than 25% missing values
misval = combined.isnull().sum()/len(combined)
combined_mv = combined.drop(columns=misval[misval > 0.25].index)
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
train = pd.concat([X, y], axis=1)
del combined_encoded, X

train.sort_values('TransactionDT', inplace=True)

X = train.drop(['isFraud'], axis=1)
y = train['isFraud']

X_train = X.iloc[:int(len(X)*.8)]
y_train = y.iloc[:int(len(X)*.8)]
X_val = X.iloc[int(len(X)*.8):]
y_val = y.iloc[int(len(X)*.8):]

## Randomized search
random_grid = {
    #'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
    'max_depth' : [int(x) for x in np.linspace(1, 30, num=6)],
    'min_child_weight' : [ 1, 3, 5, 7 ],
    'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
    'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ]
    }

# XGBoost
clf = xgb.XGBClassifier()
xg = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                         n_iter=100, cv=3, verbose=1, random_state=42, n_jobs=-1)
xg.fit(X_train, y_train)
pprint(xg.best_params_)

# Evaluation metrics
y_pred = xg.predict_proba(X_val)
auc = roc_auc_score(y_val, y_pred[:, 1])
print("Validation AUC =", auc)

acc = xg.score(X_val, y_val)
print("Accuracy: ", acc)

loss = log_loss(y_val,y_pred[:,1])
print("Log loss: ", loss)

#Confusion matrix
y_pred_bin = (y_pred > 0.5)
y_pred_bin = np.argmax(y_pred_bin, axis=1)
ConfusionMatrixDisplay.from_predictions(y_val, y_pred_bin)
plt.tight_layout()
plt.savefig('XGBConfusionMatrix.pdf', format='pdf')
plt.show()

# Roc plot
fpr, tpr, _ = roc_curve(y_val, y_pred[::,1])
plt.plot([0,1], [0,1],"k--")
plt.plot(fpr,tpr, marker='.')
plt.xlabel("False Pos Rate")
plt.ylabel("True Pos Rate")
plt.title("ROC-curve")
plt.tight_layout()
plt.savefig('ROC-curve.pdf', format='pdf')
plt.show()
