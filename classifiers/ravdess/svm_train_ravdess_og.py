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
  print(feature_idxs)
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  x = rows[:, feature_idxs]
  y = rows[:, -1]
  clf = SVC(C=10, gamma=0.001)
  clf.fit(x, y)
  s = pickle.dumps(clf)
  return s

def predict_val(file_name, feature_idxs, model):
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)

  x = rows[:, feature_idxs]
  y_true = rows[:, -1]
  clf = pickle.loads(model)
  y_pred = clf.predict(x)

  f1 = metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
  precision = metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
  recall = metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
  confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5,6,7,8])

  return f1, precision, recall, confusion_matrix

# count number of each emotion, index = label (ignore index 0)
def emotions(file_name):
  rows = np.genfromtxt(file_name,delimiter=',')
  emotions = [0] * 9
  for row_idx in range(1, len(rows)):
    y_i = int(rows[row_idx, -1])
    emotions[y_i] += 1
  return emotions

def correlation_acoustic(file_name):
  header = np.genfromtxt(file_name, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  for i in range(-13, -2):
    corr = stats.pearsonr(rows[:, i], rows[:, -1])
    print(header[i], "corr:", corr)

# returns MFCC features that should be removed
def correlation_mfcc(file_name):
  header = np.genfromtxt(file_name, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  remove_features = []
  mfcc_idxs = []
  for col_idx in range(len(header)):
    if 'mfcc' in header[col_idx]:
      mfcc_idxs.append(col_idx)
  for col in mfcc_idxs:
    corr = stats.pearsonr(rows[:, col], rows[:, -1])
    if abs(corr[0]) < 0.2:
      remove_features.append(header[col])
  return remove_features

def random_forest(file_name):
  header = np.genfromtxt(file_name, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  x = rows[:, 1:-4]
  y = rows[:, -1]

  clf = RandomForestClassifier(n_estimators=10000, random_state=1000, n_jobs=-1)
  clf.fit(x, y)

  # Print the name and gini importance of each feature
  # for feature in zip(header, clf.feature_importances_):
  #   print(feature)

  sfm = SelectFromModel(clf, threshold=-np.inf, max_features=10)
  sfm.fit(x, y)
  selected_features = []
  for feature_list_index in sfm.get_support(indices=True):
    selected_features.append(header[feature_list_index])
  return selected_features

def lda(file_name):
  header = np.genfromtxt(file_name, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(file_name, delimiter=',', skip_header=1)
  x = rows[:, 1:-4]
  y = rows[:, -1]
  clf = LinearDiscriminantAnalysis()
  clf.fit(x, y)
  print(clf.explained_variance_ratio_)

def print_results(train_file, val_file, feature_idxs):
  model = svm(train_file, feature_idxs)
  f1, precision, recall, confusion_matrix = predict_val(val_file, feature_idxs, model)
  print('f1:', f1)
  print('precision:', precision)
  print('recall:', recall)
  print(confusion_matrix)

def main(feature_type='', gender='all'):
  if gender == 'female':
    train_file = "../../final_data/ravdess/speech_train_data_numerized_female.csv"
    val_file = "../../final_data/ravdess/speech_val_data_numerized_female.csv"
  elif gender == 'male':
    train_file = "../../final_data/ravdess/speech_train_data_numerized_male.csv"
    val_file = "../../final_data/ravdess/speech_val_data_numerized_male.csv"
  else:
    train_file = "../../final_data/ravdess/speech_train_data_numerized.csv"
    val_file = "../../final_data/ravdess/speech_val_data_numerized.csv"

  # print("train:", emotions(train_file))
  # print("val:", emotions(val_file))

  header = np.genfromtxt(train_file, delimiter=',', max_rows=1, dtype='<U64')
  rows = np.genfromtxt(train_file, delimiter=',', skip_header=1)

  acoustic = [-12, -11, -9, -8, -7, -6, -5]
  remove_mfcc = correlation_mfcc(train_file)

  if feature_type == 'ma': # mfcc + acoustic
    mfcc_idxs = []
    for col_idx in range(len(header)):
      if 'mfcc' in header[col_idx] and header[col_idx] not in remove_mfcc:
        mfcc_idxs.append(col_idx)
    feature_idxs = acoustic + mfcc_idxs
    print_results(train_file, val_file, feature_idxs)

  elif feature_type == 'mfcc': # mfcc
    mfcc_idxs = []
    for col_idx in range(len(header)):
      if 'mfcc' in header[col_idx] and header[col_idx] not in remove_mfcc:
        mfcc_idxs.append(col_idx)
    feature_idxs = mfcc_idxs.copy()
    print_results(train_file, val_file, feature_idxs)

  elif feature_type == 'acoustic': # acoustic
    feature_idxs = acoustic.copy()
    print_results(train_file, val_file, feature_idxs)

  elif feature_type == 'rf': # random forest
    feature_idxs = []
    selected_features = random_forest(train_file)
    for col_idx in range(len(header)):
      if header[col_idx] in selected_features:
        feature_idxs.append(col_idx)
    print_results(train_file, val_file, feature_idxs)

  # use features from tukey's test
  elif feature_type == 'corr':
    feature_idxs = []
    if gender == 'male':
      feature_file = "../../analysis/tukey_m_features.csv"
    elif gender == 'female':
      feature_file = "../../analysis/tukey_f_features.csv"
    else: # both
      feature_file = "../../analysis/tukey_b_features.csv"

    with open(feature_file, 'r') as csvfile:
      reader = csv.reader(csvfile, delimiter='\n')
      next(reader, None)
      for row in reader:
        feature_idxs.append(np.where(header == row)[0][0])
    print_results(train_file, val_file, feature_idxs)

  elif feature_type == 'lda':
    lda(train_file)

  else: 
    feature_idxs = range(0, len(header))
    print_results(train_file, val_file, feature_idxs)

if __name__ == "__main__":
  # Usage: python svm_train_ravdess.py [mfcc / acoustic / ma / rf / corr / all / lda] [male / female / both]
  # Command line arg is MFCC / acoustic / MFCC+acoustic
  if len(sys.argv) > 1:
    main(sys.argv[1], sys.argv[2])
  else:
    main()
