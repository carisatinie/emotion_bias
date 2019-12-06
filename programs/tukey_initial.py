import csv
import pandas as pd
import argparse
import scipy.stats as ss
import os
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
import sys
import numpy as np

emotions = [1, 2, 3, 4, 5]

def get_dataframe(input_data, comparison_list, features, dataset):
    df_list = []
    for level in comparison_list:
        # looking for all rows with that emotion
        for i in range(len(input_data[input_data["emotion"]==level])):
            if dataset == "msp":
                df_list.append([input_data[input_data["emotion"]==level].loc[:, 'name'].iloc[i], level] + list(input_data[input_data["emotion"]==level].iloc[i])[2:-5] + list(input_data[input_data["emotion"]==level].iloc[i])[-3:])
            elif dataset == "ravdess":
                df_list.append([input_data[input_data["emotion"]==level].loc[:, 'name'].iloc[i], level] + list(input_data[input_data["emotion"]==level].iloc[i])[2:-4])
            elif dataset == "masc":
                df_list.append([input_data[input_data["emotion"]==level].loc[:, 'name'].iloc[i], level] + list(map(float, input_data[input_data["emotion"]==level].iloc[i][1:10])) + list(map(float, input_data[input_data["emotion"]==level].iloc[i][11:-2])))

    return pd.DataFrame(df_list, columns=["name", "emotion"]+features)

def comparison(input_csv, gender, dataset):
    input_data = pd.read_csv(input_csv)

    df = None
    columns = list(input_data.columns)
    features = columns[2:-4]
    if dataset == "msp":
        features = columns[2:-2]
        if gender == "f":
            input_data = input_data[input_data["gender"]==0]
        elif gender == "m":
            input_data = input_data[input_data["gender"]==1]
    elif dataset == "masc":
        features = columns[1:10] + columns[11:-2]
        if gender == "f":
            input_data = input_data[input_data["gender"]=="female"]
        elif gender == "m":
            input_data = input_data[input_data["gender"]=="male"]

    # argument 2 is groups we compare
    # argument 3 is one single column which we choose for values
    df = get_dataframe(input_data, emotions, features, dataset)

    # print(df['emotion'])

    frames = []
    for feature in features:
        MultiComp = MultiComparison(df[feature], df['emotion'])
        result = MultiComp.tukeyhsd()
        result_df = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])
        
        # label the block
        data = np.array([feature, 0, 0, 0, 0, 0, 0])
        dfObj = pd.DataFrame(data=[data], columns=result._results_table.data[0])
        
        frames.append(dfObj)
        frames.append(result_df)
    
    concat = pd.concat(frames)
    concat.to_csv("../analysis/new_" + dataset + "_tukey_" + gender + ".csv")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputcsv", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gender", required=True)
    parsed_args = parser.parse_args()

    comparison(parsed_args.inputcsv, parsed_args.gender, parsed_args.dataset)