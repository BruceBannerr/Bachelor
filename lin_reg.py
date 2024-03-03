import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from clean import clean

test, train = clean()
'''
train = train.fillna(-999)
test = test.fillna(-999)

y_train = train['isFraud']
X_train = train.drop(columns=['isFraud'])
X_test = test

# Label Encoding for categorical variables.
for f in X_train.columns:
    if X_train[f].dtype=='object' or test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        test[f] = lbl.transform(list(test[f].values))

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
'''
'''
train = train.dropna(how='all')
test = test.dropna(how='all')
y_train = train['isFraud']
x_train = train.drop(columns=['isFraud'])
y_test = test['isFraud']
x_test = test.drop(columns=['isFraud'])
'''
X = train.drop(["isFraud"],axis = 1)
Y = train["isFraud"]


x_train, x_val, y_train, y_val = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

Model = LogisticRegression()
Model.fit(X,Y)

print(classification_report(y_test, Model.predict(x_test), target_names=["Class 0","Class 1"]))