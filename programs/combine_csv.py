import csv

# we combine all our features together into one csv

# is09 = open("../initial_data/IS09_ravdess.csv")
# acoustic = open("../initial_data/all_features_ravdess.csv")
# output = open("../initial_data/concat_all_features_ravdess.csv", "w")
is09 = open("../initial_data/IS09_augmented_msp.csv")
acoustic = open("../initial_data/all_features_msp.csv")
output = open("../initial_data/concat_all_features_msp.csv", "w")

file_dict = {}
features = ["min_intensity", "max_intensity", "mean_intensity", "min_pitch", "max_pitch", "mean_pitch", "hnr", "jitter", "shimmer"]

# reader_1 = csv.reader(is09, delimiter=';')
reader_1 = csv.reader(is09, delimiter=',')
reader_2 = csv.reader(acoustic, delimiter=',')
writer = csv.writer(output, delimiter=',')

line_count = 0
for row in reader_2:
    if line_count == 0:
        line_count += 1
    else:
        file_dict[row[0]] = row[1:]

for row in reader_1:
    if line_count == 1:
        writer.writerow(row[1:] + features)
        line_count += 1
    else:
        try:
            writer.writerow(row[1:] + file_dict[row[1]])
        except KeyError:
            pass