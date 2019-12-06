import csv
import pandas as pd
import argparse

def analysis(input_csv, output_csv):
    input_data = open(input_csv)
    input_reader = csv.reader(input_data, delimiter=',')

    best_features = []

    columns = []
    line_count = 0
    rows = []
    feature_to_groups = {}
    feature = None

    for row in input_reader:
        if line_count == 0:
            line_count += 1
            columns = row
        else:
            if row[2] == '0':
                line_count += 1
                if rows:
                    feature_to_groups[feature] = []
                    for entry in rows:
                        if entry[6] == "True":
                            feature_to_groups[feature].append((entry[0], entry[1]))
                    rows = []
                    feature = row[1]
            else:
                rows.append(row[1:])

    # threshold - over 5
    for key in feature_to_groups:
        if len(feature_to_groups[key]) > 5:
            best_features.append(key)

    output = open(output_csv, "w")
    output_writer = csv.writer(output, delimiter=',')

    for i, best_feature in enumerate(best_features):
        output_writer.writerow([best_feature])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputcsv", required=True)
    parser.add_argument("--outputcsv", required=True)
    parsed_args = parser.parse_args()

    analysis(parsed_args.inputcsv, parsed_args.outputcsv)