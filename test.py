import pandas as pd 

testid = pd.read_csv("test_identity.csv") 
testtr = pd.read_csv("test_transaction.csv") 
trainid = pd.read_csv("train_identity.csv") 
traintr = pd.read_csv("train_transaction.csv") 

testidsmall = testid.sample(n = 500) 
testtrsmall = testtr.sample(n = 500) 
trainidsmall = trainid.sample(n = 500) 
traintrsmall = traintr.sample(n = 500) 

testidsmall.to_csv(r"/mnt/schier/Bachelor/test_id_small.csv")
testtrsmall.to_csv(r"/mnt/schier/Bachelor/test_tr_small.csv")
trainidsmall.to_csv(r"/mnt/schier/Bachelor/train_id_small.csv")
traintrsmall.to_csv(r"/mnt/schier/Bachelor/train_tr_small.csv")