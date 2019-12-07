#!/bin/bash

# copy masc data into the masc_redos folder for further processing

redos=()
input="masc_redos.txt"
while read line;
do
    redos+=($line)
done < $input

dir="../speech_datasets/man_aff_spch/data"
find "../speech_datasets/man_aff_spch/data" -type f
while read filename;
do 
    for i in "${redos[@]}"
    do
        echo "here"
        if [$i == $filename]
        then
            echo "HERE"
            cp "${dir} $filename" "masc_redos/"
        fi
    done
done

exit