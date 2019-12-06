import csv
import random
import math
import sys

initial_file = open("../final_data/masc/all_masc_data_final.csv")
initial_reader = csv.reader(initial_file, delimiter=',')

train = open("../final_data/masc/train_masc.csv", "w")
train_writer = csv.writer(train, delimiter=',')
val = open("../final_data/masc/val_masc.csv", "w")
val_writer = csv.writer(val, delimiter=',')
test = open("../final_data/masc/test_masc.csv", "w")
test_writer = csv.writer(test, delimiter=',')

list_of_stuff = []
males = ["001", "003", "005", "006", "007", "009",
        "010", "014", "016", "017", "018", "020",
        "021", "022", "023", "024", "025", "026", 
        "028", "029", "030", "032", "033", "034",
        "035", "036", "037", "040", "041", "042",
        "043", "044", "045", "046", "051", "055",
        "056", "057", "058", "060", "062", "064",
        "065", "067", "068", "069", "071", "072"]

line_count = 0
for row in initial_reader:
    if line_count == 0:
        train_writer.writerow(row + ["gender"])
        val_writer.writerow(row + ["gender"])
        test_writer.writerow(row + ["gender"])
        line_count += 1
    else:
        is_male = False
        for male in males:
            if male in row[0]:
                is_male = True
        try:
            row.remove('')
        except:
            pass
        if is_male:
            list_of_stuff.append(row + ["male"])
        else:
            list_of_stuff.append(row + ["female"])

# take 60% of list for train
train_sample = random.sample(list_of_stuff, k=math.floor(len(list_of_stuff)*6/10))

# delete from list_of_stuff
for sample in train_sample:
    train_writer.writerow(sample)
    list_of_stuff.remove(sample)

val_sample = random.sample(list_of_stuff, k=math.floor(len(list_of_stuff)*1/2))

for sample in val_sample:
    val_writer.writerow(sample)
    list_of_stuff.remove(sample)

for sample in list_of_stuff:
    test_writer.writerow(sample)