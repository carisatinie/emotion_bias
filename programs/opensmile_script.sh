#bin/bash

echo 'name' >> column_msp.csv
#for f in $(find ../speech_datasets/man_aff_spch -type f -name '*.wav'); 
for f in $(find ../speech_datasets/msp_podcast/Audio -type f -name '*.wav');
do 
SMILExtract -C ../Junior_Year/opensmile-2.3.0/config/IS09_emotion.conf -I $f -csvoutput IS09_msp.csv ; 
echo $f >> column_msp.csv;
done;
