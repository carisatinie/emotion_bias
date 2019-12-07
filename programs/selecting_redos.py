# if anything in the row is 100000000.0, then we output the name of the file, to know we have to redo
# the acoustic analysis

import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--newcsv", required=True)
parsed_args = parser.parse_args()

count = 0
with open(parsed_args.csv) as csv_file:
    with open(parsed_args.newcsv, "w") as csv_writer_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_writer = csv.writer(csv_writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                csv_writer.writerow(row)
                line_count += 1
            else:
                if '100000000.0' in row[1:]:
                    count += 1
                    csv_writer.writerow(row)

print(count)