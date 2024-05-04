from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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

#
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

# START COPY PASTED CODE https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# Randomized search https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=11)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 30, num=6)]
#max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [4, 10, 50]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 8]
# Method of selecting samples for training each tree
bootstrap = [True, False]  # Create the random grid

criterion = ['gini', 'entropy']

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterion}

# Random Forest Classifier
rf = RandomForestClassifier()
rfc = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                         n_iter=100, cv=3, verbose=1, random_state=42, n_jobs=-1)
rfc.fit(X_train, y_train)
pprint(rfc.best_params_)
# END COPY PASTED CODE

# Validation AUC
y_pred = rfc.predict_proba(X_val)
auc = roc_auc_score(y_val, y_pred[:, 1])
print("Validation AUC =", auc)

acc = rfc.score(X_val, y_val)
print("Accuracy: ", acc)

loss = log_loss(y_val,y_pred[:,1])
print("Log loss: ", loss)

#Confusion matrix
y_pred_bin = (y_pred > 0.5)
y_pred_bin = np.argmax(y_pred_bin, axis=1)
ConfusionMatrixDisplay.from_predictions(y_val, y_pred_bin)
plt.tight_layout()
plt.savefig('RFConfusionMatrix.pdf', format='pdf')
plt.show()

# Roc plot
fpr, tpr, _ = roc_curve(y_val, y_pred[::,1])
plt.plot([0,1], [0,1],"k--")
plt.plot(fpr,tpr, marker='.')
plt.xlabel("False Pos Rate")
plt.ylabel("True Pos Rate")
plt.title("ROC-curve")
plt.tight_layout()
plt.savefig('RFROC-curve.pdf', format='pdf')
plt.show()

# Feature importances
#pd.Series(rfc.feature_importances_, index=X.columns).nlargest(
    #15).plot(kind='barh')
#plt.show()

# Prediction submission
pred = rfc.predict_proba(test)
submission['isFraud'] = pred[:, 1]
