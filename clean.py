import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer
import seaborn as sns

sns.set_palette("PiYG")

# Load dataset
test_id = pd.read_csv("test_identity.csv")
test_tr = pd.read_csv("test_transaction.csv")
train_id = pd.read_csv("train_identity.csv")
train_tr = pd.read_csv("train_transaction.csv")

# Merge data
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
#del test_id, test_tr, train_id, train_tr

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
train2 = pd.concat([X,y], axis = 1)
del combined_encoded, X

train2.sort_values('TransactionDT', inplace=True)

X = train2.drop(['isFraud'], axis=1)
y = train2['isFraud']

X_train = X.iloc[:int(len(X)*.8)]
y_train = y.iloc[:int(len(X)*.8)]
X_val = X.iloc[int(len(X)*.8):]
y_val = y.iloc[int(len(X)*.8):]

#### Grrafer
e = train["isFraud"].value_counts().values
lab = ["Not Fraud", "Fraud"]
targetplot = sns.barplot(e)
targetplot.set_xticklabels(lab)
plt.title("Target variable count")
plt.savefig("target.pdf")
plt.show()

del e, lab, targetplot

# TransactionAmt (https://www.kaggle.com/code/jesucristo/fraud-complete-eda#TransactionAmt)
## Fraud == 1 subset
train_fraud = train[train["isFraud"] == 1]
fig, ax = plt.subplots(1, 2, figsize=(15, 15))

sns.distplot(np.log(train['TransactionAmt'].values), ax=ax[0])
ax[0].set_title('log Distribution of TransactionAmt')
ax[1].set_xlim([min(np.log(train['TransactionAmt'].values)), max(np.log(train['TransactionAmt'].values))])

sns.distplot(np.log(train_fraud['TransactionAmt'].values), ax=ax[1])
ax[1].set_title('log Distribution of TransactionAmt (isFraud == 1)')
ax[1].set_xlim([min(np.log(train_fraud['TransactionAmt'].values)), max(np.log(train_fraud['TransactionAmt'].values))])
plt.savefig("trans.pdf")
plt.show()

sns.countplot(x="ProductCD", hue = "isFraud", data=train)
plt.title('Count of productcode')
plt.savefig("productcd.pdf")
plt.show()

sns.countplot(x="DeviceType", data=train_id)
plt.title('Count of devicetypes')
plt.savefig("device.pdf")
plt.show()

sns.countplot(x="card4", hue = "isFraud", data=train_tr)
plt.title('Card category')
plt.savefig("card4.pdf")
plt.show()

plt.figure(figsize=(15,10))
sns.countplot(y="P_emaildomain", hue = "isFraud", data=train_tr)
plt.title('Count of purchaser emaildomain')
plt.savefig("pemail.pdf")
plt.show()
