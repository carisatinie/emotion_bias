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
import FisherExact

'''
MSP: statistical tests to determine gender bias.
chi-squared test for each emotion and feature set, comparing between male and female.
Ex: MFCC angry female vs MFCC angry male
'''

# Run SVM on given file with given features
# C and gamma are hyperparameters
def svm(file_name, feature_idxs, C, gamma):
  # print(feature_idxs)
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  x = rows[:, feature_idxs]
  y = rows[:, -1]
  clf = SVC(C=C, gamma=gamma)
  clf.fit(x, y)
  s = pickle.dumps(clf)
  return s

# file_name is for validation or test sets
def predict(file_name, feature_idxs, model):
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)

  x = rows[:, feature_idxs]
  y_true = rows[:, -1]
  clf = pickle.loads(model)
  y_pred = clf.predict(x)

  confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5])
  return confusion_matrix

# return model that was trained
def get_model(train_file, feature_idxs, C, gamma):
  model = svm(train_file, feature_idxs, C, gamma)
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
  clf = RandomForestClassifier(n_estimators=10000, random_state=1000, n_jobs=-1)
  clf.fit(x, y)
  sfm = SelectFromModel(clf, threshold=-np.inf, max_features=10)
  sfm.fit(x, y)
  rf_idxs = []
  for feature_list_index in sfm.get_support(indices=True):
    rf_idxs.append(feature_list_index)
  return rf_idxs

# runs chi squared on each emotion in the selected feature set.
# Writes results to a CSV.
def chi_squared_per_emotion(male_confusion, female_confusion, writer, feature_type):
  emotions = ["neutral", "happy", "sad", "surprised", "angry"]

  for i in range(len(male_confusion)): # for each emotion
    male_counts = male_confusion[i]
    female_counts = female_confusion[i]
    cont_table = np.stack((male_counts, female_counts))
    p = None
    if feature_type == "a":
      chisq, p, dof, exp = stats.chi2_contingency(cont_table)
    else:
      p = FisherExact.fisher_exact(cont_table)
    writer.writerow([feature_type + "_m_" + emotions[i], male_counts, feature_type + "_f_" + emotions[i], female_counts, p])

def main(feature_type, writer):
  train_female = "../../final_data/msp/fixed_data/train_data_female.csv"
  val_female = "../../final_data/msp/fixed_data/val_data_female.csv"
  test_female = "../../final_data/msp/fixed_data/test_data_female.csv"
  feature_female = "../../analysis/msp_tukey_f_features.csv"
  
  train_male = "../../final_data/msp/fixed_data/train_data_male.csv"
  val_male = "../../final_data/msp/fixed_data/val_data_male.csv"
  test_male = "../../final_data/msp/fixed_data/test_data_male.csv"
  feature_male = "../../analysis/msp_tukey_m_features.csv"

  train_both = "../../final_data/msp/fixed_data/train_data_dup.csv"
  feature_both = "../../analysis/msp_tukey_b_features.csv"

  header_female = np.genfromtxt(train_female, delimiter=',', max_rows=1, dtype='<U64')
  rows_female = np.genfromtxt(train_female, delimiter=',', skip_header=1)
  header_male = np.genfromtxt(train_female, delimiter=',', max_rows=1, dtype='<U64')
  rows_male = np.genfromtxt(train_female, delimiter=',', skip_header=1)
  header_both = np.genfromtxt(train_both, delimiter=',', max_rows=1, dtype='<U64')
  rows_both = np.genfromtxt(train_both, delimiter=',', skip_header=1)

  feature_idxs_female = []
  feature_idxs_male = []
  feature_idxs_both = []

  val_model_female = None
  val_model_male = None
  val_model_both = None

  # Retrieves appropriate feature indices. Trains model using testing and validation data.
  if feature_type == 'corr':
    with open(feature_female, 'r') as csvfile:
      reader = csv.reader(csvfile, delimiter='\n')
      next(reader, None)
      for row in reader:
        feature_idxs_female.append(np.where(header_female == row)[0][0])
    with open(feature_male, 'r') as csvfile:
      reader = csv.reader(csvfile, delimiter='\n')
      next(reader, None)
      for row in reader:
        feature_idxs_male.append(np.where(header_male == row)[0][0])
    with open(feature_both, 'r') as csvfile:
      reader = csv.reader(csvfile, delimiter='\n')
      next(reader, None)
      for row in reader:
        feature_idxs_both.append(np.where(header_both == row)[0][0])
    # validation
    val_model_female = get_model(train_female, feature_idxs_female, 10, 0.1)
    val_model_male = get_model(train_male, feature_idxs_male, 10, 0.01)
    val_model_both = get_model(train_both, feature_idxs_both, 20, 0.0001)
  elif feature_type == 'mfcc':
    feature_idxs_female = get_mfcc_features(train_female)
    feature_idxs_male = get_mfcc_features(train_male)
    feature_idxs_both = get_mfcc_features(train_both)
    # validation
    val_model_female = get_model(train_female, feature_idxs_female, 20, 0.0000001)
    val_model_male = get_model(train_male, feature_idxs_male, 10, 0.000001)
    val_model_both = get_model(train_both, feature_idxs_both, 20, 0.000001)
  elif feature_type == 'a':
    feature_idxs_female = get_acoustic_features(train_female)
    feature_idxs_male = get_acoustic_features(train_male)
    feature_idxs_both = get_acoustic_features(train_both)
    # validation
    val_model_female = get_model(train_female, feature_idxs_female, 25, 0.01)
    val_model_male = get_model(train_male, feature_idxs_male, 20, 0.01)
    val_model_both = get_model(train_both, feature_idxs_both, 10, 0.01)
  elif feature_type == 'ma':
    feature_idxs_female = get_mfcc_acoustic_features(train_female)
    feature_idxs_male = get_mfcc_acoustic_features(train_male)
    feature_idxs_both = get_mfcc_acoustic_features(train_both)
    # validation
    val_model_female = get_model(train_female, feature_idxs_female, 10, 0.000001)
    val_model_male = get_model(train_male, feature_idxs_male, 18, 0.000001)
    val_model_both = get_model(train_both, feature_idxs_both, 18, 0.000001)
  elif feature_type == 'rf':
    feature_idxs_female = get_random_forest_features(train_female)
    feature_idxs_male = get_random_forest_features(train_male)
    feature_idxs_both = get_random_forest_features(train_both)
    # validation
    val_model_female = get_model(train_female, feature_idxs_female, 20, 0.00001)
    val_model_male = get_model(train_male, feature_idxs_male, 10, 0.01)
    val_model_both = get_model(train_both, feature_idxs_both, 20, 0.001)

  # Run tuned model on test data. Get confusion matrix for each gender, and run chi squared test.
  confusion_female = predict(test_female, feature_idxs_female, val_model_both)
  confusion_male = predict(test_male, feature_idxs_male, val_model_both)

  chi_squared_per_emotion(confusion_male, confusion_female, writer, feature_type)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run chi-squared with MSP.')
  parser.add_argument('--feature_type')
  parsed_args = parser.parse_args()

  # main(parsed_args.feature_type)
  chi_squared_file = open("msp_chisquared.csv", "a+")
  chi_squared_writer = csv.writer(chi_squared_file, delimiter=',')
  # chi_squared_writer.writerow(["group1", "value1", "group2", "value2", "fisher_test"])

  # main("corr", chi_squared_writer)
  # print('corr done')
  # main("mfcc", chi_squared_writer)
  # print('mfcc done')
  main("a", chi_squared_writer)
  print('a done')
  main("ma", chi_squared_writer)
  print('ma done')
  main("rf", chi_squared_writer)
  print('rf done')
