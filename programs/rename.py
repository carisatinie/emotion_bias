import csv

# we rename the names of the files for the features

with open('all_features_masc_1.csv') as csv_file:
    with open('all_features_masc.csv', "w") as csv_write:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_writer = csv.writer(csv_write, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                csv_writer.writerow(row)
                line_count += 1
            else:
                # masc
                name = row[0][:-8] + "/" + row[0][-8:]

                # msp
                # name = row[0][:36] + "/" + row[0][36:]

                # ravdess
                # name = row[0][:35] + "/" + row[0][35:]

                csv_writer.writerow([name] + row[1:])