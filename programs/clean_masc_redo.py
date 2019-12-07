import csv

# we needed to rerun a Praat script on the wav files that Parselmouth
# couldn't process, so we clean out the files we need to rerun here

masc = open("../masc_redo_txt/redos.csv")
masc_reader = csv.reader(masc, delimiter=',')
initial = open("../initial_data/all_features_masc.csv")
initial_reader = csv.reader(initial, delimiter=',')

masc_w = open("../final_data/masc/all_masc_data.csv", "w")
masc_writer = csv.writer(masc_w, delimiter=',')

line_count = 0
for row in initial_reader:
    if line_count == 0:
        masc_writer.writerow(row)
        line_count += 1
    else:
        if 'nan' not in row and '100000000.0' not in row:
            masc_writer.writerow(row)

for row in masc_reader:
    if line_count == 1:
        line_count += 1
    else:
        if row[0] != "Filename":
            masc_writer.writerow(row)

masc.close()
initial.close()
masc_w.close()