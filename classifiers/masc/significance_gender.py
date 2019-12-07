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
MASC: statistical tests to determine feature set bias.
chi-squared test for  each pair of feature sets, with the same gender
'''
rf_final = None
# Run SVM on given file with given features
# C and gamma are hyperparameters
def svm(file_name, feature_idxs, C, gamma):
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
  acoustic_idxs = range(-13, -4)
  return acoustic_idxs

# Return indices of acoustic and MFCC features from the dataset. Makes sure that no duplicates are returned.
def get_mfcc_acoustic_features(file_name):
  mfcc_idxs = get_mfcc_features(file_name)
  acoustic_idxs = get_acoustic_features(file_name)
  return list(set().union(mfcc_idxs, acoustic_idxs))

# Returns indices of features obtained from using Random Forest feature selection.
# Runs random forest and gathers the 10 most important features.
# Optimized to only run when necessary to increase efficiency
def get_random_forest_features(file_name):
  global rf_final
  if not rf_final:
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
    rf_final = rf_idxs
    print(rf_idxs)
    return rf_idxs
  else:
    return rf_final

# Given two arrays of consideration, creates a contigency table.
# Runs the fisher exact test, a variation of chi squared test for smaller tables.
# Writes results to a CSV.
def chi_squared(confusion1, confusion2, gender, params1, params2):
  chi_squared_file = open("masc_chisquared_gender.csv", "a+")
  writer = csv.writer(chi_squared_file, delimiter=',')
  emotions = ["neutral", "happy", "sad", "surprised", "angry"]
  for i in range(len(confusion1)): # for each emotion
    counts1 = confusion1[i]
    counts2 = confusion2[i]
    cont_table = np.stack((counts1, counts2))
    p = None
    if params1 == "corr" and params2 == "a" or params1 == "a" and params2 == "m" or params1 == "a" and params2 == "m+a" or params1 == "m+a" and params2 == "rf": # doesn't need for male
      chisq, p, dof, exp = stats.chi2_contingency(cont_table)
    else:
      p = FisherExact.fisher_exact(cont_table)

    if gender == "female":
      writer.writerow([params1 + "_f_" + emotions[i], counts1, params2 + "_f_" + emotions[i], counts2, p])
    else:
      writer.writerow([params1 + "_m_" + emotions[i], counts1, params2 + "_m_" + emotions[i], counts2, p])

  chi_squared_file.close()

# Finds every pair of feature sets within male and within female. 
# Runs fisher exact test using the confusion matrices returned by SVM.
def main(feature_type):
  train_female = "../../final_data/masc/train_female.csv"
  val_female = "../../final_data/masc/val_female.csv"
  test_female = "../../final_data/masc/test_female.csv"
  feature_female = "../../analysis/new_masc_tukey_f_features.csv"

  train_male = "../../final_data/masc/train_male.csv"
  val_male = "../../final_data/masc/val_male.csv"
  test_male = "../../final_data/masc/test_male.csv"
  feature_male = "../../analysis/new_masc_tukey_m_features.csv"

  header_female = np.genfromtxt(train_female, delimiter=',', max_rows=1, dtype='<U64')
  rows_female = np.genfromtxt(train_female, delimiter=',', skip_header=1)
  header_male = np.genfromtxt(train_female, delimiter=',', max_rows=1, dtype='<U64')
  rows_male = np.genfromtxt(train_female, delimiter=',', skip_header=1)

  # FEMALE / FEMALE
  # [Tukey's, acoustic, mfcc, mfcc+acoustic, random forest]
  # Tuned hyperparameters. Each tuple corresponds to one feature set in the order: Tukey's, acoustic, MFCC, MFCC+acoustic, random forest.
  parameters = [(10, 0.000001), (20, 0.00001), (20, 0.000001), (18, 0.00001), (18, 0.000001)]
  params = ["corr", "a", "m", "m+a", "rf"]

  for i in range(5):
    for j in range(i, 5):
      print((i, j))
      feature_idxs_1 = []
      feature_idxs_2 = []
      C_1 = parameters[i][0]
      gamma_1 = parameters[i][1]
      C_2 = parameters[j][0]
      gamma_2 = parameters[j][1]

      # first feature set
      if i == 0: # Tukey's
        with open(feature_female, 'r') as csvfile:
          reader = csv.reader(csvfile, delimiter='\n')
          next(reader, None)
          for row in reader:
            feature_idxs_1.append(np.where(header_female == row)[0][0])
      elif i == 1: # acoustic
        feature_idxs_1 = get_acoustic_features(train_female)
      elif i == 2: # mfcc
        feature_idxs_1 = get_mfcc_features(train_female)
      elif i == 3: # mfcc+acoustic
        feature_idxs_1 = get_mfcc_acoustic_features(train_female)
      else: # random forest
        feature_idxs_1 = get_random_forest_features(train_female)

      # second feature set
      if j == 0: # Tukey's
        with open(feature_female, 'r') as csvfile:
          reader = csv.reader(csvfile, delimiter='\n')
          next(reader, None)
          for row in reader:
            feature_idxs_2.append(np.where(header_female == row)[0][0])
      elif j == 1: # acoustic
        feature_idxs_2 = get_acoustic_features(train_female)
      elif j == 2: # mfcc
        feature_idxs_2 = get_mfcc_features(train_female)
      elif j == 3: # mfcc+acoustic
        feature_idxs_2 = get_mfcc_acoustic_features(train_female)
      else: # random forest
        feature_idxs_2 = get_random_forest_features(train_female)

      # Get trained models for each feature set
      val_model_1 = get_model(train_female, feature_idxs_1, C_1, gamma_1)
      val_model_2 = get_model(train_female, feature_idxs_2, C_2, gamma_2)

      # Predict on the test data and obtain confusion matrices
      confusion_1 = predict(test_female, feature_idxs_1, val_model_1)
      confusion_2 = predict(test_female, feature_idxs_2, val_model_2)

      chi_squared(confusion_1, confusion_2, "female", params[i], params[j])

  # MALE / MALE
  # [Tukey's, acoustic, mfcc, mfcc+acoustic, random forest]
  # Same as above, except for males instead of females.
  parameters = [(10, 0.000001), (10, 0.0001), (10, 0.000001), (20, 0.000001), (18, 0.00001)]
  global rf_final
  rf_final = None

  for i in range(3, 5):
    for j in range(i, 5):
      print((i, j))
      run = True
      if i == 3:
        if j < 4:
          run = False
      if run:
        feature_idxs_1 = []
        feature_idxs_2 = []
        C_1 = parameters[i][0]
        gamma_1 = parameters[i][1]
        C_2 = parameters[j][0]
        gamma_2 = parameters[j][1]

        # first feature set
        if i == 0: # Tukey's
          with open(feature_male, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\n')
            next(reader, None)
            for row in reader:
              feature_idxs_1.append(np.where(header_male == row)[0][0])
        elif i == 1: # acoustic
          feature_idxs_1 = get_acoustic_features(train_male)
        elif i == 2: # mfcc
          feature_idxs_1 = get_mfcc_features(train_male)
        elif i == 3: # mfcc+acoustic
          feature_idxs_1 = get_mfcc_acoustic_features(train_male)
        else: # random forest
          feature_idxs_1 = get_random_forest_features(train_male)

        # second feature set
        if j == 0: # Tukey's
          with open(feature_male, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\n')
            next(reader, None)
            for row in reader:
              feature_idxs_2.append(np.where(header_male == row)[0][0])
        elif j == 1: # acoustic
          feature_idxs_2 = get_acoustic_features(train_male)
        elif j == 2: # mfcc
          feature_idxs_2 = get_mfcc_features(train_male)
        elif j == 3: # mfcc+acoustic
          feature_idxs_2 = get_mfcc_acoustic_features(train_male)
        else: # random forest
          feature_idxs_2 = get_random_forest_features(train_male)

        # validation
        val_model_1 = get_model(train_male, feature_idxs_1, C_1, gamma_1)
        val_model_2 = get_model(train_male, feature_idxs_2, C_2, gamma_2)

        confusion_1 = predict(test_male, feature_idxs_1, val_model_1)
        confusion_2 = predict(test_male, feature_idxs_2, val_model_2)

        chi_squared(confusion_1, confusion_2, "male", params[i], params[j])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run chi-squared with MASC.')
  parser.add_argument('--feature_type')
  parsed_args = parser.parse_args()

  chi_squared_file = open("masc_chisquared_gender.csv", "a+")
  chi_squared_writer = csv.writer(chi_squared_file, delimiter=',')

  main("corr")