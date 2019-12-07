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

# Run SVM on a given file (train, test, or val) and with certain feature indices that are found 
# in main(), according to the feature set type passed in as a command line argument.
def svm(file_name, feature_idxs):
  print(feature_idxs)
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  x = rows[:, feature_idxs]
  y = rows[:, -1]
  clf = SVC(C=20, gamma=0.001)
  clf.fit(x, y)
  s = pickle.dumps(clf)
  return s

# file_name is for validation or test data
# Given a model, runs that model on the given file and outputs F1 score and confusion matrix.
def predict(file_name, feature_idxs, model):
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)

  x = rows[:, feature_idxs]
  y_true = rows[:, -1]
  clf = pickle.loads(model)

  y_pred = clf.predict(x)

  # Compute scores of prediction using actual labels of given dataset.
  f1 = metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
  precision = metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
  recall = metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
  confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5])

  return f1, precision, recall, confusion_matrix

# Helper function to print results from predict(), as well as return the model used.
def print_results(train_file, val_file, feature_idxs):
  model = svm(train_file, feature_idxs)
  f1, precision, recall, confusion_matrix = predict(val_file, feature_idxs, model)
  print('f1:', f1)
  print('precision:', precision)
  print('recall:', recall)
  print(confusion_matrix)
  return model

# Return indices of MFCC features from the dataset. All MFCC features will contain the string "mfcc"
def get_mfcc_features(file_name):
  header = np.genfromtxt(file_name, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  mfcc_idxs = []
  for col_idx in range(len(header)):
    if 'mfcc' in header[col_idx]:
      mfcc_idxs.append(col_idx)
  return mfcc_idxs

# Return indices of acoustic features from the dataset. The acoustic features are known to be in the columns used here.
def get_acoustic_features(file_name):
  header = np.genfromtxt(file_name, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  acoustic_idxs = range(-14, -5)
  return acoustic_idxs

# Return indices of acoustic and MFCC features from the dataset. Makes sure that no duplicates are returned.
def get_mfcc_acoustic_features(file_name):
  mfcc_idxs = get_mfcc_features(file_name)
  acoustic_idxs = get_acoustic_features(file_name)
  return list(set().union(mfcc_idxs, acoustic_idxs))

# Returns indices of features obtained from using Random Forest feature selection.
# Runs random forest and gathers the 10 most important features.
def get_random_forest_features(file_name):
  header = np.genfromtxt(file_name, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  x = rows[:, 1:-4]
  y = rows[:, -1]

  print(header[369], header[370], header[394])

  clf = RandomForestClassifier(n_estimators=10000, random_state=1000, n_jobs=-1)
  clf.fit(x, y)
  sfm = SelectFromModel(clf, threshold=-np.inf, max_features=10)
  sfm.fit(x, y)
  rf_idxs = []
  for feature_list_index in sfm.get_support(indices=True):
    rf_idxs.append(feature_list_index)
  return rf_idxs

def main(feature_type, gender):
  # Depending on gender, grabs appropriate training, validation, and test set files.
  # Tukey's features are obtained from a separate script in the analysis folder that writes them to a CSV.
  if gender == 'female':
    train_file = "../../final_data/msp/fixed_data/train_data_female.csv"
    val_file = "../../final_data/msp/fixed_data/val_data_female.csv"
    test_file = "../../final_data/msp/fixed_data/test_data_female.csv"
    feature_file = "../../analysis/msp_tukey_f_features.csv"
  elif gender == 'male':
    train_file = "../../final_data/msp/fixed_data/train_data_male.csv"
    val_file = "../../final_data/msp/fixed_data/val_data_male.csv"
    test_file = "../../final_data/msp/fixed_data/test_data_male.csv"
    feature_file = "../../analysis/msp_tukey_m_features.csv"
  else:
    train_file = "../../final_data/msp/fixed_data/train_data_dup.csv"
    val_file = "../../final_data/msp/fixed_data/val_data_dup.csv"
    test_file = "../../final_data/msp/fixed_data/test_data_dup.csv"
    feature_file = "../../analysis/msp_tukey_b_features.csv"

  header = np.genfromtxt(train_file, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(train_file, delimiter=',', skip_header=1)

  # Based on feature type from command line, grabs appropriate feature indices to train model with.
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

  # Get trained model after validation.
  val_model = print_results(train_file, val_file, feature_idxs)

  # predictions on the test set using the model trained on training + validation sets
  print("--- TEST PREDICTION ---")
  f1, precision, recall, confusion_matrix = predict(test_file, feature_idxs, val_model)
  print('f1:', f1)
  print('precision:', precision)
  print('recall:', recall)
  print(confusion_matrix)

if __name__ == "__main__":
  # Run like so: python3 svm_train_msp.py --feature_type=[corr,a,mfcc,ma,rf] --gender=[male,female,both]
	parser = argparse.ArgumentParser(description='Run SVM with MSP.')
	parser.add_argument('--feature_type', required=True)
	parser.add_argument('--gender', required=True)
	parsed_args = parser.parse_args()

	main(parsed_args.feature_type, parsed_args.gender)
