import csv
import pandas as pd

# train = pd.read_csv("final_data/ravdess/speech_train_data.csv")
# validation = pd.read_csv("final_data/ravdess/speech_val_data.csv")
# test = pd.read_csv("final_data/ravdess/speech_test_data.csv")

# print('train f', train[train['gender']==0].count())
# print('train m', train[train['gender']==1].count())
# print('val f', validation[validation['gender']==0].count())
# print('val m', validation[validation['gender']==1].count())
# print('test f', test[test['gender']==0].count())
# print('test m', test[test['gender']==1].count())

train = pd.read_csv("final_data/msp/train_data.csv")
validation = pd.read_csv("final_data/msp/val_data.csv")
test = pd.read_csv("final_data/msp/test_data.csv")

print('train f', train[train['gender']=='Female'].count())
print('train m', train[train['gender']=='Male'].count())
print('val f', validation[validation['gender']=='Female'].count())
print('val m', validation[validation['gender']=='Male'].count())
print('test f', test[test['gender']=='Female'].count())
print('test m', test[test['gender']=='Male'].count())