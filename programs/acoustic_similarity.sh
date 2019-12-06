#!bin/bash/

for i in {1..9}
do
    echo "speaker"$i
    python3 /Users/jessicahuynh/Downloads/CorpusTools-1.4.1/corpustools/acousticsim/main.py --speaker="../masc_for_acoustic_comparison/00"$i"/" --output="../acoustic_similarity/00"$i"_extra_4.csv"
done

for i in {10..68}
do
    echo "speaker"$i
    python3 /Users/jessicahuynh/Downloads/CorpusTools-1.4.1/corpustools/acousticsim/main.py --speaker="../masc_for_acoustic_comparison/0"$i"/" --output="../acoustic_similarity/0"$i"_extra_4.csv"
done
