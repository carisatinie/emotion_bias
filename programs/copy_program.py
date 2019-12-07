import csv
import shutil
import os

# we copy wav files so we can redo the feature extraction on the failed ones

# with open("masc_redos.csv", "r") as redos:
#     csv_reader = csv.reader(redos, delimiter=",")
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             line_count += 1
#         else:
#             if not os.path.exists("masc_redos/" + row[0][37:-8]):
#                 os.makedirs("masc_redos/" + row[0][37:-8])
#             shutil.copy(row[0], "masc_redos/" + row[0][37:])

with open("../final_data/ravdess/speech_test_data.csv", "r") as data:
    csv_reader = csv.reader(data, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            shutil.copy("../" + row[0], "../final_data/ravdess/test/" + row[0][36:])