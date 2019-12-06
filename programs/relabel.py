import csv

# a = open("../final_data/ravdess/speech_test_data_numerized_female.csv")
# b = open("../final_data/ravdess/speech_test_data_numerized_male.csv")
# c = open("../final_data/ravdess/speech_test_data_numerized.csv")
# d = open("../final_data/ravdess/speech_train_data_numerized_female.csv")
# e = open("../final_data/ravdess/speech_train_data_numerized_male.csv")
# f = open("../final_data/ravdess/speech_train_data_numerized.csv")
# g = open("../final_data/ravdess/speech_val_data_numerized_female.csv")
# h = open("../final_data/ravdess/speech_val_data_numerized_male.csv")
# i = open("../final_data/ravdess/speech_val_data_numerized.csv")

a = open("../final_data/msp/test_data_dup.csv")
b = open("../final_data/msp/test_data_female.csv")
c = open("../final_data/msp/test_data_male.csv")
d = open("../final_data/msp/train_data_dup.csv")
e = open("../final_data/msp/train_data_female.csv")
f = open("../final_data/msp/train_data_male.csv")
g = open("../final_data/msp/val_data_dup.csv")
h = open("../final_data/msp/val_data_female.csv")
i = open("../final_data/msp/val_data_male.csv")

a_read = csv.reader(a, delimiter=',')
b_read = csv.reader(b, delimiter=',')
c_read = csv.reader(c, delimiter=',')
d_read = csv.reader(d, delimiter=',')
e_read = csv.reader(e, delimiter=',')
f_read = csv.reader(f, delimiter=',')
g_read = csv.reader(g, delimiter=',')
h_read = csv.reader(h, delimiter=',')
i_read = csv.reader(i, delimiter=',')

# a1 = open("../final_data/ravdess/fixed_data/speech_test_data_numerized_female.csv", "w")
# b1 = open("../final_data/ravdess/fixed_data/speech_test_data_numerized_male.csv", "w")
# c1 = open("../final_data/ravdess/fixed_data/speech_test_data_numerized.csv", "w")
# d1 = open("../final_data/ravdess/fixed_data/speech_train_data_numerized_female.csv", "w")
# e1 = open("../final_data/ravdess/fixed_data/speech_train_data_numerized_male.csv", "w")
# f1 = open("../final_data/ravdess/fixed_data/speech_train_data_numerized.csv", "w")
# g1 = open("../final_data/ravdess/fixed_data/speech_val_data_numerized_female.csv", "w")
# h1 = open("../final_data/ravdess/fixed_data/speech_val_data_numerized_male.csv", "w")
# i1 = open("../final_data/ravdess/fixed_data/speech_val_data_numerized.csv", "w")
a1 = open("../final_data/msp/fixed_data/test_data_dup.csv", "w")
b1 = open("../final_data/msp/fixed_data/test_data_female.csv", "w")
c1 = open("../final_data/msp/fixed_data/test_data_male.csv", "w")
d1 = open("../final_data/msp/fixed_data/train_data_dup.csv", "w")
e1 = open("../final_data/msp/fixed_data/train_data_female.csv", "w")
f1 = open("../final_data/msp/fixed_data/train_data_male.csv", "w")
g1 = open("../final_data/msp/fixed_data/val_data_dup.csv", "w")
h1 = open("../final_data/msp/fixed_data/val_data_female.csv", "w")
i1 = open("../final_data/msp/fixed_data/val_data_male.csv", "w")

a1_writer = csv.writer(a1, delimiter=',')
b1_writer = csv.writer(b1, delimiter=',')
c1_writer = csv.writer(c1, delimiter=',')
d1_writer = csv.writer(d1, delimiter=',')
e1_writer = csv.writer(e1, delimiter=',')
f1_writer = csv.writer(f1, delimiter=',')
g1_writer = csv.writer(g1, delimiter=',')
h1_writer = csv.writer(h1, delimiter=',')
i1_writer = csv.writer(i1, delimiter=',')

original = [a_read, b_read, c_read, d_read, e_read, f_read, g_read, h_read, i_read]
new = [a1_writer, b1_writer, c1_writer, d1_writer, e1_writer, f1_writer, g1_writer, h1_writer, i1_writer]

for i in range(len(original)):
    line_count = 0
    for row in original[i]:
        if line_count == 0:
            line_count += 1
            new[i].writerow(row)
        else:
            if row:
                new_row = row
                if new_row[-1] == "3":
                    new_row[-1] = 2
                elif new_row[-1] == "4":
                    new_row[-1] = 3
                elif new_row[-1] == "6":
                    new_row[-1] = 4
                elif new_row[-1] == "7":
                    new_row[-1] = 5
                elif new_row[-1] == "8":
                    new_row[-1] = 4
                new[i].writerow(new_row)