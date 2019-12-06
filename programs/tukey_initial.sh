#!bin/bash/

# python3 tukey_initial.py --inputcsv="../final_data/ravdess/fixed_data/speech_train_data_numerized.csv" --gender="b"  --dataset="ravdess"
# echo "done"
# python3 tukey_initial.py --inputcsv="../final_data/ravdess/fixed_data/speech_train_data_numerized_male.csv" --gender="m" --dataset="ravdess"
# echo "done"
# python3 tukey_initial.py --inputcsv="../final_data/ravdess/fixed_data/speech_train_data_numerized_female.csv" --gender="f" --dataset="ravdess"
# echo "done"
# python3 tukey_initial.py --inputcsv="../final_data/msp/fixed_data/train_data_dup.csv" --gender="f" --dataset="msp"
# echo "done"
# python3 tukey_initial.py --inputcsv="../final_data/msp/fixed_data/train_data_dup.csv" --gender="m" --dataset="msp"
# echo "done"
# python3 tukey_initial.py --inputcsv="../final_data/msp/fixed_data/train_data_dup.csv" --gender="b" --dataset="msp"
# echo "done"

python3 tukey_initial.py --inputcsv="../final_data/masc/train_masc.csv" --gender="f" --dataset="masc"
echo "done"
python3 tukey_initial.py --inputcsv="../final_data/masc/train_masc.csv" --gender="m" --dataset="masc"
echo "done"
python3 tukey_initial.py --inputcsv="../final_data/masc/train_masc.csv" --gender="b" --dataset="masc"
echo "done"