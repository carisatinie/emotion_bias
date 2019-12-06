#!bin/bash/

adjectives=('anger' 'elation' 'neutral' 'panic' 'sadness')

for i in {1..9}
do
    # go through each directory
    filepath="../../speech_datasets/man_aff_spch/data/00"$i
    mkdir "../masc_for_acoustic_comparison/00"$i

    mkdir "../masc_for_acoustic_comparison/00"$i"/neutral_neutral"
    mkdir "../masc_for_acoustic_comparison/00"$i"/neutral_panic"
    mkdir "../masc_for_acoustic_comparison/00"$i"/neutral_anger"
    mkdir "../masc_for_acoustic_comparison/00"$i"/neutral_sadness"
    mkdir "../masc_for_acoustic_comparison/00"$i"/neutral_elation"

    mkdir "../masc_for_acoustic_comparison/00"$i"/panic_neutral"
    mkdir "../masc_for_acoustic_comparison/00"$i"/panic_panic"
    mkdir "../masc_for_acoustic_comparison/00"$i"/panic_anger"
    mkdir "../masc_for_acoustic_comparison/00"$i"/panic_sadness"
    mkdir "../masc_for_acoustic_comparison/00"$i"/panic_elation"

    mkdir "../masc_for_acoustic_comparison/00"$i"/anger_neutral"
    mkdir "../masc_for_acoustic_comparison/00"$i"/anger_panic"
    mkdir "../masc_for_acoustic_comparison/00"$i"/anger_anger"
    mkdir "../masc_for_acoustic_comparison/00"$i"/anger_sadness"
    mkdir "../masc_for_acoustic_comparison/00"$i"/anger_elation"

    mkdir "../masc_for_acoustic_comparison/00"$i"/sadness_neutral"
    mkdir "../masc_for_acoustic_comparison/00"$i"/sadness_panic"
    mkdir "../masc_for_acoustic_comparison/00"$i"/sadness_anger"
    mkdir "../masc_for_acoustic_comparison/00"$i"/sadness_sadness"
    mkdir "../masc_for_acoustic_comparison/00"$i"/sadness_elation"

    mkdir "../masc_for_acoustic_comparison/00"$i"/elation_neutral"
    mkdir "../masc_for_acoustic_comparison/00"$i"/elation_panic"
    mkdir "../masc_for_acoustic_comparison/00"$i"/elation_anger"
    mkdir "../masc_for_acoustic_comparison/00"$i"/elation_sadness"
    mkdir "../masc_for_acoustic_comparison/00"$i"/elation_elation"
    for adjective in "${adjectives[@]}"
    do
        for j in {11..12}
        do
            for f in $filepath"/"$adjective"/utterance/2"$j"*.wav"
            do
                cp $f "../masc_for_acoustic_comparison/00"$i"/"$adjective"_neutral/"
            done
        done
        for f in "../masc_for_acoustic_comparison/00"$i"/"$adjective"_neutral"/*
            do
                a=${adjective:0:1}
                if [ $a == 'a' ] 
                then
                    mv "$f" "${f/.wav/_a_n.wav}"
                elif [ $a == 'n' ]
                then
                    mv "$f" "${f/.wav/_n_n.wav}"
                elif [ $a == 'e' ]
                then
                    mv "$f" "${f/.wav/_e_n.wav}"
                elif [ $a == 's' ]
                then
                    mv "$f" "${f/.wav/_s_n.wav}"
                else
                    mv "$f" "${f/.wav/_p_n.wav}"
                fi
            done
        for j in {13..14}
        do
            for f in $filepath"/"$adjective"/utterance/2"$j"*.wav"
            do
                cp $f "../masc_for_acoustic_comparison/00"$i"/"$adjective"_panic/"
            done
        done
        for f in "../masc_for_acoustic_comparison/00"$i"/"$adjective"_panic"/*
            do
                a=${adjective:0:1}
                if [ $a == 'a' ] 
                then
                    mv "$f" "${f/.wav/_a_p.wav}"
                elif [ $a == 'n' ]
                then
                    mv "$f" "${f/.wav/_n_p.wav}"
                elif [ $a == 'e' ]
                then
                    mv "$f" "${f/.wav/_e_p.wav}"
                elif [ $a == 's' ]
                then
                    mv "$f" "${f/.wav/_s_p.wav}"
                else
                    mv "$f" "${f/.wav/_p_p.wav}"
                fi
            done
        for j in {15..16}
        do
            for f in $filepath"/"$adjective"/utterance/2"$j"*.wav"
            do
                cp $f "../masc_for_acoustic_comparison/00"$i"/"$adjective"_anger/"
            done
        done
        for f in "../masc_for_acoustic_comparison/00"$i"/"$adjective"_anger"/*
            do
                a=${adjective:0:1}
                if [ $a == 'a' ] 
                then
                    mv "$f" "${f/.wav/_a_a.wav}"
                elif [ $a == 'n' ]
                then
                    mv "$f" "${f/.wav/_n_a.wav}"
                elif [ $a == 'e' ]
                then
                    mv "$f" "${f/.wav/_e_a.wav}"
                elif [ $a == 's' ]
                then
                    mv "$f" "${f/.wav/_s_a.wav}"
                else
                    mv "$f" "${f/.wav/_p_a.wav}"
                fi
            done
        for j in {17..18}
        do
            for f in $filepath"/"$adjective"/utterance/2"$j"*.wav"
            do
                cp $f "../masc_for_acoustic_comparison/00"$i"/"$adjective"_sadness/"
            done
        done
        for f in "../masc_for_acoustic_comparison/00"$i"/"$adjective"_sadness"/*
            do
                a=${adjective:0:1}
                if [ $a == 'a' ] 
                then
                    mv "$f" "${f/.wav/_a_s.wav}"
                elif [ $a == 'n' ]
                then
                    mv "$f" "${f/.wav/_n_s.wav}"
                elif [ $a == 'e' ]
                then
                    mv "$f" "${f/.wav/_e_s.wav}"
                elif [ $a == 's' ]
                then
                    mv "$f" "${f/.wav/_s_s.wav}"
                else
                    mv "$f" "${f/.wav/_p_s.wav}"
                fi
            done
        for j in {19..20}
        do
            for f in $filepath"/"$adjective"/utterance/2"$j"*.wav"
            do
                cp $f "../masc_for_acoustic_comparison/00"$i"/"$adjective"_elation/"
            done
        done
        for f in "../masc_for_acoustic_comparison/00"$i"/"$adjective"_elation"/*
            do
                a=${adjective:0:1}
                if [ $a == 'a' ] 
                then
                    mv "$f" "${f/.wav/_a_e.wav}"
                elif [ $a == 'n' ]
                then
                    mv "$f" "${f/.wav/_n_e.wav}"
                elif [ $a == 'e' ]
                then
                    mv "$f" "${f/.wav/_e_e.wav}"
                elif [ $a == 's' ]
                then
                    mv "$f" "${f/.wav/_s_e.wav}"
                else
                    mv "$f" "${f/.wav/_p_e.wav}"
                fi
            done
    done
done

for i in {10..68}
do
    # go through each directory
    filepath="../../speech_datasets/man_aff_spch/data/0"$i
    mkdir "../masc_for_acoustic_comparison/0"$i

    mkdir "../masc_for_acoustic_comparison/0"$i"/neutral_neutral"
    mkdir "../masc_for_acoustic_comparison/0"$i"/neutral_panic"
    mkdir "../masc_for_acoustic_comparison/0"$i"/neutral_anger"
    mkdir "../masc_for_acoustic_comparison/0"$i"/neutral_sadness"
    mkdir "../masc_for_acoustic_comparison/0"$i"/neutral_elation"

    mkdir "../masc_for_acoustic_comparison/0"$i"/panic_neutral"
    mkdir "../masc_for_acoustic_comparison/0"$i"/panic_panic"
    mkdir "../masc_for_acoustic_comparison/0"$i"/panic_anger"
    mkdir "../masc_for_acoustic_comparison/0"$i"/panic_sadness"
    mkdir "../masc_for_acoustic_comparison/0"$i"/panic_elation"

    mkdir "../masc_for_acoustic_comparison/0"$i"/anger_neutral"
    mkdir "../masc_for_acoustic_comparison/0"$i"/anger_panic"
    mkdir "../masc_for_acoustic_comparison/0"$i"/anger_anger"
    mkdir "../masc_for_acoustic_comparison/0"$i"/anger_sadness"
    mkdir "../masc_for_acoustic_comparison/0"$i"/anger_elation"

    mkdir "../masc_for_acoustic_comparison/0"$i"/sadness_neutral"
    mkdir "../masc_for_acoustic_comparison/0"$i"/sadness_panic"
    mkdir "../masc_for_acoustic_comparison/0"$i"/sadness_anger"
    mkdir "../masc_for_acoustic_comparison/0"$i"/sadness_sadness"
    mkdir "../masc_for_acoustic_comparison/0"$i"/sadness_elation"

    mkdir "../masc_for_acoustic_comparison/0"$i"/elation_neutral"
    mkdir "../masc_for_acoustic_comparison/0"$i"/elation_panic"
    mkdir "../masc_for_acoustic_comparison/0"$i"/elation_anger"
    mkdir "../masc_for_acoustic_comparison/0"$i"/elation_sadness"
    mkdir "../masc_for_acoustic_comparison/0"$i"/elation_elation"
    for adjective in "${adjectives[@]}"
    do
        for j in {11..12}
        do
            for f in $filepath"/"$adjective"/utterance/2"$j"*.wav"
            do
                cp $f "../masc_for_acoustic_comparison/0"$i"/"$adjective"_neutral/"
            done
        done
        for f in "../masc_for_acoustic_comparison/0"$i"/"$adjective"_neutral"/*
            do
                a=${adjective:0:1}
                if [ $a == 'a' ] 
                then
                    mv "$f" "${f/.wav/_a_n.wav}"
                elif [ $a == 'n' ]
                then
                    mv "$f" "${f/.wav/_n_n.wav}"
                elif [ $a == 'e' ]
                then
                    mv "$f" "${f/.wav/_e_n.wav}"
                elif [ $a == 's' ]
                then
                    mv "$f" "${f/.wav/_s_n.wav}"
                else
                    mv "$f" "${f/.wav/_p_n.wav}"
                fi
            done
        for j in {13..14}
        do
            for f in $filepath"/"$adjective"/utterance/2"$j"*.wav"
            do
                cp $f "../masc_for_acoustic_comparison/0"$i"/"$adjective"_panic/"
            done
        done
        for f in "../masc_for_acoustic_comparison/0"$i"/"$adjective"_panic"/*
            do
                a=${adjective:0:1}
                if [ $a == 'a' ] 
                then
                    mv "$f" "${f/.wav/_a_p.wav}"
                elif [ $a == 'n' ]
                then
                    mv "$f" "${f/.wav/_n_p.wav}"
                elif [ $a == 'e' ]
                then
                    mv "$f" "${f/.wav/_e_p.wav}"
                elif [ $a == 's' ]
                then
                    mv "$f" "${f/.wav/_s_p.wav}"
                else
                    mv "$f" "${f/.wav/_p_p.wav}"
                fi
            done
        for j in {15..16}
        do
            for f in $filepath"/"$adjective"/utterance/2"$j"*.wav"
            do
                cp $f "../masc_for_acoustic_comparison/0"$i"/"$adjective"_anger/"
            done
        done
        for f in "../masc_for_acoustic_comparison/0"$i"/"$adjective"_anger"/*
            do
                a=${adjective:0:1}
                if [ $a == 'a' ] 
                then
                    mv "$f" "${f/.wav/_a_a.wav}"
                elif [ $a == 'n' ]
                then
                    mv "$f" "${f/.wav/_n_a.wav}"
                elif [ $a == 'e' ]
                then
                    mv "$f" "${f/.wav/_e_a.wav}"
                elif [ $a == 's' ]
                then
                    mv "$f" "${f/.wav/_s_a.wav}"
                else
                    mv "$f" "${f/.wav/_p_a.wav}"
                fi
            done
        for j in {17..18}
        do
            for f in $filepath"/"$adjective"/utterance/2"$j"*.wav"
            do
                cp $f "../masc_for_acoustic_comparison/0"$i"/"$adjective"_sadness/"
            done
        done
        for f in "../masc_for_acoustic_comparison/0"$i"/"$adjective"_sadness"/*
            do
                a=${adjective:0:1}
                if [ $a == 'a' ] 
                then
                    mv "$f" "${f/.wav/_a_s.wav}"
                elif [ $a == 'n' ]
                then
                    mv "$f" "${f/.wav/_n_s.wav}"
                elif [ $a == 'e' ]
                then
                    mv "$f" "${f/.wav/_e_s.wav}"
                elif [ $a == 's' ]
                then
                    mv "$f" "${f/.wav/_s_s.wav}"
                else
                    mv "$f" "${f/.wav/_p_s.wav}"
                fi
            done
        for j in {19..20}
        do
            for f in $filepath"/"$adjective"/utterance/2"$j"*.wav"
            do
                cp $f "../masc_for_acoustic_comparison/0"$i"/"$adjective"_elation/"
            done
        done
        for f in "../masc_for_acoustic_comparison/0"$i"/"$adjective"_elation"/*
            do
                a=${adjective:0:1}
                if [ $a == 'a' ] 
                then
                    mv "$f" "${f/.wav/_a_e.wav}"
                elif [ $a == 'n' ]
                then
                    mv "$f" "${f/.wav/_n_e.wav}"
                elif [ $a == 'e' ]
                then
                    mv "$f" "${f/.wav/_e_e.wav}"
                elif [ $a == 's' ]
                then
                    mv "$f" "${f/.wav/_s_e.wav}"
                else
                    mv "$f" "${f/.wav/_p_e.wav}"
                fi
            done
    done
done