import argparse
import csv
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
from scipy import stats

def svm(file_name, feature_idxs):
  # print(feature_idxs)
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  x = rows[:, feature_idxs]
  y = rows[:, -1]
  clf = SVC(C=18, gamma=0.00001)
  clf.fit(x, y)
  s = pickle.dumps(clf)
  return s

def predict(file_name, feature_idxs, model):
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)

  x = rows[:, feature_idxs]
  y_true = rows[:, -1]
  clf = pickle.loads(model)

  y_pred = clf.predict(x)

  f1 = metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
  precision = metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
  recall = metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
  confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5])

  return f1, precision, recall, confusion_matrix

def print_results(train_file, val_file, feature_idxs):
  model = svm(train_file, feature_idxs)
  f1, precision, recall, confusion_matrix = predict(val_file, feature_idxs, model)
  print('f1:', f1)
  print('precision:', precision)
  print('recall:', recall)
  print(confusion_matrix)
  return model

def get_mfcc_features(file_name):
  header = np.genfromtxt(file_name, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  mfcc_idxs = []
  for col_idx in range(len(header)):
    if 'mfcc' in header[col_idx]:
      mfcc_idxs.append(col_idx)
  return mfcc_idxs

def get_acoustic_features(file_name):
  header = np.genfromtxt(file_name, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  acoustic_idxs = range(1, 10)
  return acoustic_idxs

def get_mfcc_acoustic_features(file_name):
  mfcc_idxs = get_mfcc_features(file_name)
  acoustic_idxs = get_acoustic_features(file_name)
  return list(set().union(mfcc_idxs, acoustic_idxs))

def get_random_forest_features(file_name):
  header = np.genfromtxt(file_name, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  x = rows[:, 1:-4]
  y = rows[:, -1]
  clf = RandomForestClassifier(n_estimators=10000, random_state=1000, n_jobs=-1)
  clf.fit(x, y)
  sfm = SelectFromModel(clf, threshold=-np.inf, max_features=10)
  sfm.fit(x, y)
  rf_idxs = []
  for feature_list_index in sfm.get_support(indices=True):
    rf_idxs.append(feature_list_index)
  return rf_idxs

def main(feature_type, gender):
  if gender == 'female':
    train_file = "../../final_data/masc/train_female.csv"
    val_file = "../../final_data/masc/val_female.csv"
    test_file = "../../final_data/masc/test_female.csv"
    feature_file = "../../analysis/new_masc_tukey_f_features.csv"
  elif gender == 'male':
    train_file = "../../final_data/masc/train_male.csv"
    val_file = "../../final_data/masc/val_male.csv"
    test_file = "../../final_data/masc/test_male.csv"
    feature_file = "../../analysis/new_masc_tukey_m_features.csv"
  else:
    train_file = "../../final_data/masc/train_masc_2.csv"
    val_file = "../../final_data/masc/val_masc_2.csv"
    test_file = "../../final_data/masc/test_masc_2.csv"
    feature_file = "../../analysis/new_masc_tukey_b_features.csv"

  header = np.genfromtxt(train_file, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(train_file, delimiter=',', skip_header=1)

  feature_idxs = []
  if feature_type == 'corr':
  	with open(feature_file, 'r') as csvfile:
  	  reader = csv.reader(csvfile, delimiter='\n')
  	  next(reader, None)
  	  for row in reader:
  	  	feature_idxs.append(np.where(header == row)[0][0])
  elif feature_type == 'mfcc':
  	feature_idxs = get_mfcc_features(train_file)
  elif feature_type == 'a':
  	feature_idxs = get_acoustic_features(train_file)
  elif feature_type == 'ma':
  	feature_idxs = get_mfcc_acoustic_features(train_file)
  elif feature_type == 'rf':
  	feature_idxs = get_random_forest_features(train_file)

  # validation
  val_model = print_results(train_file, val_file, feature_idxs)

  # predict test set
  print("--- TEST PREDICTION ---")
  f1, precision, recall, confusion_matrix = predict(test_file, feature_idxs, val_model)
  print('f1:', f1)
  print('precision:', precision)
  print('recall:', recall)
  print(confusion_matrix)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run SVM with MASC.')
	parser.add_argument('--feature_type', required=True)
	parser.add_argument('--gender', required=True)
	parsed_args = parser.parse_args()

	main(parsed_args.feature_type, parsed_args.gender)
