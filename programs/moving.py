import csv
import pandas as pd

# replace first column of IS09_masc.csv with first column of column.csv
masc = pd.read_csv('IS09_msp.csv', sep=";")
column = pd.read_csv('column_msp.csv', sep=";")

masc['name'] = list(column['name'])

masc.to_csv('IS09_augmented_msp.csv')