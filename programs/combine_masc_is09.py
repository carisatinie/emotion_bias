import csv
import sys

# we combine all our features together for IS09 and take out rows with --undefined--

new_file = open("../initial_data/IS09_augmented_masc.csv")
reader = csv.reader(new_file, delimiter=',')

# all_data = open("../final_data/masc/all_masc_data.csv")
# reader2 = csv.reader(all_data, delimiter=',')

# augmented_data = open("../final_data/masc/all_masc_data_augmented.csv", "w")
# writer = csv.writer(augmented_data, delimiter=',')

# line_count = 0
# for row in reader2:
#     if line_count == 0:
#         line_count += 1
#         writer.writerow(row)
#     else:
#         if "/Users/jessicahuynh/Desktop/neutrality/masc_redos/" in row[0]:
#             new_name = row[0].split("/")[-1][-8:]
#             phrase = row[0].split("/")[-1][:-8]
#             name = "/".join(row[0].split("/")[:-1]) + "/" + phrase + "/" + new_name

#             writer.writerow([name] + row[1:])
#         else:
#             writer.writerow(row)

augmented_data = open("../final_data/masc/all_masc_data_augmented.csv")
reader2 = csv.reader(augmented_data, delimiter=',')

new_output = open("../final_data/masc/all_masc_data_final.csv", "w")
writer = csv.writer(new_output, delimiter=',')

row_from_data = []
line_count = 0
name_to_data = {}
for row in reader2:
    if line_count == 0:
        line_count += 1
        row_from_data = row
    else:
        name = "/".join(row[0].split("/")[-4:])
        name_to_data[name] = list(map(str, row[1:]))

for row in reader:
    if line_count == 1:
        line_count += 1
        row_from_data += row[2:]
        writer.writerow(row_from_data + ["emotion"])
    else:
        name = "/".join(row[1].split("/")[-4:])
        try:
            output = [name] + name_to_data[name] + row[2:]
            if "neutral" in name:
                output += [1]
            elif "elation" in name:
                output += [2]
            elif "sadness" in name:
                output += [3]
            elif "panic" in name:
                output += [4]
            else:
                output += [5]

            for i in range(len(output)):
                if type(output[i]) == type("val"):
                    string = output[i]
                    output[i] = string.replace(' ', '')

            if "--undefined--" in output:
                pass
            else:
                writer.writerow(output)
                line_count += 1
        except KeyError:
            pass