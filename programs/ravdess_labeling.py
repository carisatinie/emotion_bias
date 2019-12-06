import csv

with open('../initial_data/concat_all_features_ravdess.csv') as csv_file:
    with open('../final_data/labeled_features_ravdess.csv', "w") as csv_writer_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_writer = csv.writer(csv_writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                csv_writer.writerow(row + ['speech_song', 'normal_strong', 'gender', 'emotion'])
                line_count += 1
            else:
                name_of_file = list(row[0][-24:-4].split("-"))
                additional_info = [name_of_file[1], name_of_file[3], str(int(name_of_file[6]) % 2)]
                if name_of_file[2] == "01":
                    additional_info.append("neutral")
                elif name_of_file[2] == "02":
                    additional_info.append("calm")
                elif name_of_file[2] == "03":
                    additional_info.append("happy")
                elif name_of_file[2] == "04":
                    additional_info.append("sad")
                elif name_of_file[2] == "05":
                    additional_info.append("angry")
                elif name_of_file[2] == "06":
                    additional_info.append("fearful")
                elif name_of_file[2] == "07":
                    additional_info.append("disgust")
                else:
                    additional_info.append("surprised")
                csv_writer.writerow(row + additional_info)
