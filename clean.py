import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer
import seaborn as sns

sns.set_palette("bright")

# Load dataset
test_id = pd.read_csv("test_identity.csv")
test_tr = pd.read_csv("test_transaction.csv")
train_id = pd.read_csv("train_identity.csv")
train_tr = pd.read_csv("train_transaction.csv")

# Merge data
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
del test_id, test_tr, train_id, train_tr

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
plt.savefig("target.pdf", format="pdf")
plt.show()

# Hvor mange er der egentlig af hver
f = train["isFraud"].value_counts()
print(f)

del e, f, lab, targetplot

# Transaction fraud fraction
## Fraud == 1 subset
train_fraud = train[train["isFraud"] == 1]
fig, ax = plt.subplots(1, 2, figsize=(15, 15))

# Transaction amount
sns.distplot(np.log(train['TransactionAmt'].values), ax=ax[0])
ax[0].set_title('log Distribution of TransactionAmt')
ax[1].set_xlim([min(np.log(train['TransactionAmt'].values)), max(np.log(train['TransactionAmt'].values))])

sns.distplot(np.log(train_fraud['TransactionAmt'].values), ax=ax[1])
ax[1].set_title('log Distribution of TransactionAmt (isFraud == 1)')
ax[1].set_xlim([min(np.log(train_fraud['TransactionAmt'].values)), max(np.log(train_fraud['TransactionAmt'].values))])
plt.savefig("trans.pdf", format="pdf")
plt.show()

# Product code countplot
sns.countplot(x="ProductCD", hue = "isFraud", data=train)
plt.title('Count of productcode')
plt.savefig("productcd.pdf", format="pdf")
plt.show()

# DeviceType plot
sns.countplot(x="DeviceType", data=train_id)
plt.title('Count of devicetypes')
plt.savefig("device.pdf", format="pdf")
plt.show()

# Card category
sns.countplot(x="card4", hue = "isFraud", data=train_tr)
plt.title('Card category')
plt.savefig("card4.pdf", format="pdf")
plt.show()

# Email domain
plt.figure(figsize=(15,10))
sns.countplot(y="P_emaildomain", hue = "isFraud", data=train_tr, , order=train_tr.P_emaildomain.value_counts().iloc[:6].index)
plt.title('Count of purchaser emaildomain')
plt.savefig("pemail.pdf", format="pdf")
plt.show()

# fraud and time of the day
# inspired from: https://www.kaggle.com/code/fchmiel/day-and-time-powerful-predictive-feature/notebook
def make_hour_feature(df, tname='TransactionDT'):
    hours = df[tname] / (3600)        
    encoded_hours = np.floor(hours) % 24
    return encoded_hours

train['hours'] = make_hour_feature(train)

plt.plot(train.groupby('hours')['isFraud'].mean(), color='k')
ax = plt.gca()
ax2 = ax.twinx()
_ = ax2.hist(train['hours'], alpha=0.3, bins=24)
ax.set_xlabel('Encoded hour')
ax.set_ylabel('Fraction of fraudulent transactions')

ax2.set_ylabel('Number of transactions')
plt.tight_layout()
plt.savefig("hour.pdf", format="pdf")
plt.show()
