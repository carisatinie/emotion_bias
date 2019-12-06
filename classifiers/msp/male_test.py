import numpy as np

train_file = "../../final_data/msp/fixed_data/train_data_male.csv"
val_file = "../../final_data/msp/fixed_data/val_data_male.csv"
test_file = "../../final_data/msp/fixed_data/test_data_male.csv"

header = np.genfromtxt(test_file, delimiter=',', max_rows=1, dtype='<U64')
rows = np.genfromtxt(test_file, delimiter=',', skip_header=1)
print(rows.shape)

print(header[393])

for row_idx in range(len(rows)):
	# print('.')
	# print(row)
	row = rows[row_idx]
	res = np.isnan(row[1:]).any()

	if res:
		print(res)
		print(row_idx)
		print(np.argwhere(np.isnan(row[1:])))
	if row[394] > 1000:
		print(row[394])