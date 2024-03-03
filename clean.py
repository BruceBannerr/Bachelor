import pandas as pd 
import math


def clean(printing=False):
    test_id = pd.read_csv("test_identity.csv")
    test_tr = pd.read_csv("test_transaction.csv")
    train_id = pd.read_csv("train_identity.csv")
    train_tr = pd.read_csv("train_transaction.csv")
    
    if (printing):
        # Hvor mange procent i hver kolonne er NaN?
        print("_____test id______________________")
        print(test_id.isnull().mean() * 100)
        print("_____test tr______________________")
        print(test_tr.isnull().mean() * 100)
        print("_____train id______________________")
        print(train_id.isnull().mean() * 100)
        print("_____train tr______________________")
        print(train_tr.isnull().mean() * 100)
    

    if (printing):
    #Sorter dem fra hvor vi mangler mere end 25% af data
        print("_______________test id____________________")
        print(test_id.shape)
    test_id_thresh = math.floor(test_id.shape[0]*0.75)
    test_id = test_id.dropna(axis='columns', thresh = test_id_thresh)
    if (printing):
        print(test_id.shape)
        print("_______________test tr____________________")
        print(test_tr.shape)
    test_tr_thresh = math.floor(test_tr.shape[0]*0.75)
    test_tr = test_tr.dropna(axis='columns', thresh = test_tr_thresh)
    if (printing):
        print(test_tr.shape)
        print("_______________train id____________________")
        print(train_id.shape)
    train_id_thresh = math.floor(train_id.shape[0]*0.75)
    train_id = train_id.dropna(axis='columns', thresh = train_id_thresh)
    if (printing):
        print(train_id.shape)
        print("_______________train tr____________________")
        print(train_tr.shape)
    train_tr_thresh = math.floor(train_tr.shape[0]*0.75)
    train_tr = train_tr.dropna(axis='columns', thresh = train_tr_thresh)
    if (printing):
        print(train_tr.shape)
    

    ###############################Vi kan nok ikke bare lige gøre det, da nogle kommentarer siger at ID ikke helt passer
    train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
    test = pd.merge(test_tr, test_id, on='TransactionID', how='left')

    return test, train


if __name__ == "__main__":
    clean(printing=True)
