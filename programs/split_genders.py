import argparse

''' 
Given a file with both genders, creates 2 CSVs, one for male and one for female.
Uses 'gender' column; '0'=female, '1'=male.
'''
def split_genders(file, dest_file, gender_idx):
  lines = open(file, 'r').readlines()

  with open(dest_file + '_male.csv','w') as file:
    for line_idx in range(len(lines)):
      if line_idx == 0:
        file.write(lines[0])
        file.write('\n')
      else:
        line = lines[line_idx]
        line_list = line.split(',')
        gender = line_list[gender_idx].strip()
        if gender == '1' or gender == 'male':
          file.write(line)
          file.write('\n')

  with open(dest_file + '_female.csv','w') as file:
    for line_idx in range(len(lines)):
      if line_idx == 0:
        file.write(lines[0])
        file.write('\n')
      else:
        line = lines[line_idx]
        line_list = line.split(',')
        gender = line_list[gender_idx].strip()
        if gender == '0' or gender == 'female':
          file.write(line)
          file.write('\n')

if __name__ == "__main__":
  # Usage: python3 split_genders.py --data=[ravdess,msp,masc]
  parser = argparse.ArgumentParser(description='Split data into male and female CSVs.')
  parser.add_argument('--data', required=True)
  data = parser.parse_args().data

  if data == 'ravdess':
    train_file = "final_data/ravdess/speech_train_data_numerized.csv"
    val_file = "final_data/ravdess/speech_val_data_numerized.csv"
    test_file = "final_data/ravdess/speech_test_data_numerized.csv"
    split_genders(train_file, "final_data/ravdess/speech_train_data_numerized", -2)
    split_genders(val_file, "final_data/ravdess/speech_val_data_numerized", -2)
    split_genders(test_file, "final_data/ravdess/speech_test_data_numerized", -2)

  elif data == 'msp':
    train_file = "final_data/msp/train_data_dup.csv"
    val_file = "final_data/msp/val_data_dup.csv"
    test_file = "final_data/msp/test_data_dup.csv"
    split_genders(train_file, "final_data/msp/train_data", -2)
    split_genders(val_file, "final_data/msp/val_data", -2)
    split_genders(test_file, "final_data/msp/test_data", -2)

  elif data == 'masc':
    train_file = "final_data/masc/train_masc_2.csv"
    val_file = "final_data/masc/val_masc_2.csv"
    test_file = "final_data/masc/test_masc_2.csv"
    split_genders(train_file, "final_data/masc/train", -2)
    split_genders(val_file, "final_data/masc/val", -2)
    split_genders(test_file, "final_data/masc/test", -2)

  
