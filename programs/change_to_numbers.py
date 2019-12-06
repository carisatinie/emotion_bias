import csv

train_data = open("../final_data/ravdess/speech_train_data.csv")
val_data = open("../final_data/ravdess/speech_val_data.csv")
test_data = open("../final_data/ravdess/speech_test_data.csv")

train_output = open("../final_data/ravdess/speech_train_data_numerized.csv", "w")
val_output = open("../final_data/ravdess/speech_val_data_numerized.csv", "w")
test_output = open("../final_data/ravdess/speech_test_data_numerized.csv", "w")

train_reader = csv.reader(train_data, delimiter=',')
val_reader = csv.reader(val_data, delimiter=',')
test_reader = csv.reader(test_data, delimiter=',')

train_writer = csv.writer(train_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
val_writer = csv.writer(val_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
test_writer = csv.writer(test_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

emotion_dict = {'neutral': 1, 'calm': 2, 'happy': 3, 'sad': 4, 'angry': 5, 'fearful': 6, 'disgust': 7, 'surprised': 8}

line_number = 0
for row in train_reader:
    if line_number == 0:
        line_number += 1
        train_writer.writerow(row)
    else:
        train_writer.writerow(row[:-1] + [emotion_dict[row[-1]]])

for row in val_reader:
    if line_number == 1:
        line_number += 1
        val_writer.writerow(row)
    else:
        val_writer.writerow(row[:-1] + [emotion_dict[row[-1]]])

for row in test_reader:
    if line_number == 2:
        line_number += 1
        test_writer.writerow(row)
    else:
        test_writer.writerow(row[:-1] + [emotion_dict[row[-1]]])