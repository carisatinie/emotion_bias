import csv

# create the csv file with all information for msp

train = 0
test = 0
validation = 0
other = 0
train_1 = 0
test_1 = 0
validation_1 = 0

speaker_to_gender = {}
file_to_gender = {}
file_to_tvt = {}
with open("../../speech_datasets/msp_podcast/Speaker_ids_2.txt", "r") as ids:
	for id in ids:
		speaker_to_gender[id.split("; ",1)[0][8:]] = id.split("; ",1)[1][:-1]

with open('../../speech_datasets/msp_podcast/Partitions.txt', "r") as partition:
		for line in partition:
			if line == "\n":
				pass
			else:
				file_to_tvt[line.split("; ",1)[1][:-1]] = line.split("; ",1)[0]

with open("../../speech_datasets/msp_podcast/Speaker_ids_1.txt", "r") as ids:
	for id in ids:
		stripped = id.split("; ",1)[1].strip('\n')
		if stripped != "Unknown":
			file_to_gender[id.split("; ",1)[0]] = speaker_to_gender[stripped]
		else:
			if file_to_tvt[id.split("; ",1)[0]] == "Train":
				train += 1
			elif file_to_tvt[id.split("; ",1)[0]] == "Validation":
				validation += 1
			else:
				test += 1

file_to_components = {}
with open("../../speech_datasets/msp_podcast/labels.txt", "r") as labels:
	look_at_next_line = False
	for label in labels:
		if not look_at_next_line:
			if label == "\n":
				look_at_next_line = True
		else:
			components = label.strip("\n").split(";")
			components = [components[i].strip(" ") for i in range(len(components))]
			file_name = components[0]
			try:
				gender = file_to_gender[file_name]
				if gender == "Female":
					gender = 0
				else:
					gender = 1
				emotion = components[1]
				if emotion == "A":
					emotion = 5
				elif emotion == "S":
					emotion = 4
				elif emotion == "H":
					emotion = 3
				elif emotion == "U":
					emotion = 8
				elif emotion == "F":
					emotion = 6
				elif emotion == "D":
					emotion = 7
				elif emotion == "C":
					emotion = 9
				elif emotion == "N":
					emotion = 1
				else:
					emotion = "X"
					if file_to_tvt[file_name] == "Train":
						train_1 += 1
					elif file_to_tvt[file_name] == "Validation":
						validation_1 += 1
					else:
						test_1 += 1
					other += 1
				if emotion != "X":
					activation = components[2][2:]
					valence = components[3][2:]
					dominance = components[4][2:]
					file_to_components[file_name] = [emotion, activation, valence, dominance, gender]
			except:
				pass

all_data = open("../initial_data/concat_all_features_msp.csv", "r")
csv_reader = csv.reader(all_data, delimiter=',')

train_file = open("../final_data/msp/train_data.csv", "w")
train_writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
validate_file = open("../final_data/msp/val_data.csv", "w")
validate_writer = csv.writer(validate_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
test_file = open("../final_data/msp/test_data.csv", "w")
test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

line_count = 0
for row in csv_reader:
	if line_count == 0:
		train_writer.writerow(row + ['emotion', 'activation', 'valence', 'dominance', 'gender'])
		validate_writer.writerow(row + ['emotion', 'activation', 'valence', 'dominance', 'gender'])
		test_writer.writerow(row + ['emotion', 'activation', 'valence', 'dominance', 'gender'])
		line_count += 1
	else:
		try:
			if file_to_tvt[row[0][37:]] == "Train":
				train_writer.writerow(row + file_to_components[row[0][37:]])
			elif file_to_tvt[row[0][37:]] == "Validation":
				validate_writer.writerow(row + file_to_components[row[0][37:]])
			else:
				test_writer.writerow(row + file_to_components[row[0][37:]])
		except:
			pass