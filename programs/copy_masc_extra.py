import csv

# we combine all our acoustic similarity calculation files together
# we ran several iterations to get more information

for i in range(1, 69):
    file_name = None
    if i < 10:
        file_name = "00" + str(i)
    else:
        file_name = "0" + str(i)

    old_file = open("../acoustic_similarity/" + file_name + ".csv")
    old_file_reader = csv.reader(old_file, delimiter=',')

    new_file = open("../acoustic_similarity/" + file_name + "_full.csv", "w")
    new_file_writer = csv.writer(new_file, delimiter=',')

    line_count = 0
    for row in old_file_reader:
        new_file_writer.writerow(row)

    extras_file = open("../acoustic_similarity/" + file_name + "_extra.csv")
    extras_file_reader = csv.reader(extras_file, delimiter=',')

    for row in extras_file_reader:
        if line_count == 0:
            line_count += 1
        else:
            new_file_writer.writerow(row)

    extras2_file = open("../acoustic_similarity/" + file_name + "_extra_2.csv")
    extras2_file_reader = csv.reader(extras2_file, delimiter=',')

    for row in extras2_file_reader:
        if line_count == 1:
            line_count += 1
        else:
            new_file_writer.writerow(row)

    extras3_file = open("../acoustic_similarity/" + file_name + "_extra_3.csv")
    extras3_file_reader = csv.reader(extras3_file, delimiter=',')

    for row in extras3_file_reader:
        if line_count == 2:
            line_count += 1
        else:
            new_file_writer.writerow(row)

    extras4_file = open("../acoustic_similarity/" + file_name + "_extra_4.csv")
    extras4_file_reader = csv.reader(extras4_file, delimiter=',')

    for row in extras4_file_reader:
        if line_count == 3:
            line_count += 1
        else:
            new_file_writer.writerow(row)