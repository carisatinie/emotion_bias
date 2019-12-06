import csv
import glob
import sys
import pandas as pd
import scipy.stats as ss
import numpy as np
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

def emotion_comparison(emotion, emotions, anova, failed):
    # want to know if it failed tukey

    # is neutral_neutral closer to neutral_anger than anger_anger
    comparison_one = []
    for second_emotion in emotions:
        if second_emotion != emotion:
            comparison_one.append(emotion + "_" + second_emotion)
    
    comparison_one_results = []

    anova_file = pd.read_csv(anova)
    for comparison in comparison_one:
        one = comparison.split("_")[0] + "_" + comparison.split("_")[0]
        two = comparison.split("_")[1] + "_" + comparison.split("_")[1]
        # neutral_anger, row that has the comparison
        df = anova_file[(anova_file['comparing_var'] == comparison) & (anova_file['pair_1'] == one) & (anova_file['pair_2'] == two)]

        if df.shape[0] > 1:
            df = df.iloc[0]
        if list(df['p_value'])[0] < 0.05:
            is_failed = False
            for fails in failed:
                if fails[0] == comparison and fails[1] == one and fails[2] == two or fails[0] == comparison and fails[1] == two and fails[2] == one:
                    is_failed = True

            if list(df['comparison'])[0] < 0:
                if is_failed:
                    comparison_one_results.append(('True', 'Failed', list(df['comparison'])[0], comparison))
                else:
                    comparison_one_results.append(('True', 'NotFailed', list(df['comparison'])[0], comparison))
            else:
                if is_failed:
                    comparison_one_results.append(('False', 'Failed', list(df['comparison'])[0], comparison))
                else:
                    comparison_one_results.append(('False', 'NotFailed', list(df['comparison'])[0], comparison))

    return comparison_one_results

def emotion_comparison_tukey(emotion, emotions, tukey, failed, anova, speaker):
    tukey_file = pd.read_csv(tukey)
    anova_file = pd.read_csv(anova)

    # Is neutral_anger different from neutral_neutral, neutral_elation, neutral_sadness, neutral_panic
    # running ANOVA between neutral_neutral/neutral_neutral and neutral_neutral/neutral_anger
    
    comparison_two = []
    comparison_two_results = []
    total = 0
    for second_emotion in emotions:
        comparison_two.append(emotion + "_" + second_emotion)

    for i in range(len(comparison_two)):
        for j in range(i, len(comparison_two)):
            if i != j:
                group1 = comparison_two[i] + "/" + comparison_two[i]
                group2 = [comparison_two[i] + "/" + comparison_two[j], comparison_two[j] + "/" + comparison_two[i]]

                comparison = None
                combo1 = None
                combo2 = None

                df = None
                for group in group2:
                    df = tukey_file[(tukey_file["group1"] == group1) & (tukey_file["group2"] == group)]
                    if df.empty:
                        df = tukey_file[(tukey_file["group1"] == group) & (tukey_file["group2"] == group1)]
                        if not df.empty:
                            combo1 = group
                            combo2 = group1
                    else:
                        combo1 = group1
                        combo2 = group

                if df.empty:
                    df = tukey_file[(tukey_file["group1"]==combo1) & (tukey_file["group2"]==combo2)]

                if df.shape[0] > 1:
                    df = df.iloc[0]
                
                if combo1.split("/")[0] == combo2.split("/")[0]:
                    comparison = combo1.split("/")[0]
                    combo1 = combo1.split("/")[1]
                    combo2 = combo2.split("/")[1]
                else:
                    comparison = combo1.split("/")[1]
                    combo1 = combo1.split("/")[0]
                    combo2 = combo2.split("/")[0]

                is_failed = False
                for fails in failed:
                    if fails[0] == comparison and fails[1] == combo1 and fails[2] == combo2 or fails[0] == comparison and fails[1] == combo2 and fails[2] == combo1:
                        is_failed = True

                # run a tukey
                if list(df['reject'])[0] and not is_failed:
                    #anova_file

                    df_compare = list(anova_file[anova_file["adjs"] == (combo1 + "/" + combo2)]["similarity"])
                    if not df_compare:
                        df_compare = list(anova_file[anova_file["adjs"] == (combo2 + "/" + combo1)]["similarity"])
                    
                    df_oneone = list(anova_file[anova_file["adjs"] == (combo1 + "/" + combo1)]["similarity"])
                    df_twotwo = list(anova_file[anova_file["adjs"] == (combo2 + "/" + combo2)]["similarity"])

                    labels = np.append(np.full(len(df_compare), 0), [np.full(len(df_oneone), 1), np.full(len(df_twotwo), 2)])
                    data = np.array(df_compare + df_oneone + df_twotwo)

                    new_df = pd.DataFrame(list(zip(labels, data)), columns=["labels", "values"])

                    # now do tukey
                    MultiComp = MultiComparison(new_df["values"], new_df["labels"])
                    result = MultiComp.tukeyhsd()
                    new_tukey_df = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])
                    new_tukey_df.to_csv("../acoustic_similarity_tukey/tukey_results/" + speaker + "_" + combo1 + "_" + combo2 + "_tukey_two.csv")

                    # find which ones are true
                    for index, row in new_tukey_df.iterrows():
                        a = None
                        b = None
                        if row['reject']:
                            if row['group1'] == 0:
                                a = combo1 + combo2
                            elif row['group1'] == 1:
                                a = combo1
                            else:
                                a = combo2
                            if row['group2'] == 0:
                                b = combo1 + combo2
                            elif row['group2'] == 1:
                                b = combo1
                            else:
                                b = combo2
                            comparison_two_results.append(a + " different than " + b)

    # for i in range(len(comparison_two)):
    #     for j in range(len(comparison_two)):
    #         for k in range(len(comparison_two)):
    #             if i != j and i != k and j != k:
    #                 group1 = [comparison_two[i] + "/" + comparison_two[j], comparison_two[j] + "/" + comparison_two[i]]
    #                 group2 = [comparison_two[i] + "/" + comparison_two[k], comparison_two[k] + "/" + comparison_two[i]]

    #                 possible_combos = []
    #                 for group in group1:
    #                     for group_2 in group2:
    #                         possible_combos.append([group, group_2])

    #                 comparison = None
    #                 combo1 = None
    #                 combo2 = None

    #                 df = None
    #                 for combo in possible_combos:
    #                     df = tukey_file[(tukey_file["group1"] == combo[0]) & (tukey_file["group2"] == combo[1])]
    #                     if df.empty:
    #                         df = tukey_file[(tukey_file["group1"] == combo[1]) & (tukey_file["group2"] == combo[0])]
    #                         if not df.empty:
    #                             combo1 = combo[1]
    #                             combo2 = combo[0]
    #                             break
    #                     else:
    #                         combo1 = combo[0]
    #                         combo2 = combo[1]
    #                         break
                    
    #                 if df.shape[0] > 1:
    #                     df = df.iloc[0]
                    
    #                 if combo1.split("/")[0] == combo2.split("/")[0]:
    #                     comparison = combo1.split("/")[0]
    #                     combo1 = combo1.split("/")[1]
    #                     combo2 = combo2.split("/")[1]
    #                 else:
    #                     comparison = combo1.split("/")[1]
    #                     combo1 = combo1.split("/")[0]
    #                     combo2 = combo2.split("/")[0]

    #                 is_failed = False
    #                 for fails in failed:
    #                     if fails[0] == comparison and fails[1] == combo1 and fails[2] == combo2 or fails[0] == comparison and fails[1] == combo2 and fails[2] == combo1:
    #                         is_failed = True

    #                 if list(df['reject'])[0] and not is_failed:
    #                     comparison_two_results += 1

    #                 total += 1

    # comparison_two_percent = comparison_two_results/total

    # Is neutral_anger different from anger_anger, elation_anger, sadness_anger, panic_anger
    comparison_three = []
    comparison_three_results = []

    for second_emotion in emotions:
        comparison_three.append(second_emotion + "_" + emotion)

    for i in range(len(comparison_three)):
        for j in range(i, len(comparison_three)):
            if i != j:
                group1 = comparison_three[i] + "/" + comparison_three[i]
                group2 = [comparison_three[i] + "/" + comparison_three[j], comparison_three[j] + "/" + comparison_three[i]]

                comparison = None
                combo1 = None
                combo2 = None

                df = None
                for group in group2:
                    df = tukey_file[(tukey_file["group1"] == group1) & (tukey_file["group2"] == group)]
                    if df.empty:
                        df = tukey_file[(tukey_file["group1"] == group) & (tukey_file["group2"] == group1)]
                        if not df.empty:
                            combo1 = group
                            combo2 = group1
                    else:
                        combo1 = group1
                        combo2 = group

                if df.empty:
                    df = tukey_file[(tukey_file["group1"]==combo1) & (tukey_file["group2"]==combo2)]

                if df.shape[0] > 1:
                    df = df.iloc[0]
                
                if combo1.split("/")[0] == combo2.split("/")[0]:
                    comparison = combo1.split("/")[0]
                    combo1 = combo1.split("/")[1]
                    combo2 = combo2.split("/")[1]
                else:
                    comparison = combo1.split("/")[1]
                    combo1 = combo1.split("/")[0]
                    combo2 = combo2.split("/")[0]

                is_failed = False
                for fails in failed:
                    if fails[0] == comparison and fails[1] == combo1 and fails[2] == combo2 or fails[0] == comparison and fails[1] == combo2 and fails[2] == combo1:
                        is_failed = True

                # run a tukey
                if list(df['reject'])[0] and not is_failed:
                    #anova_file

                    # if combo1/combo2 (0) similarity values are different than combo1/combo1 values (if they are,
                    # combo1 and combo2 are different)
                    df_compare = list(anova_file[anova_file["adjs"] == (combo1 + "/" + combo2)]["similarity"])
                    if not df_compare:
                        df_compare = list(anova_file[anova_file["adjs"] == (combo2 + "/" + combo1)]["similarity"])
                    
                    df_oneone = list(anova_file[anova_file["adjs"] == (combo1 + "/" + combo1)]["similarity"])
                    df_twotwo = list(anova_file[anova_file["adjs"] == (combo2 + "/" + combo2)]["similarity"])

                    labels = np.append(np.full(len(df_compare), 0), [np.full(len(df_oneone), 1), np.full(len(df_twotwo), 2)])
                    data = np.array(df_compare + df_oneone + df_twotwo)

                    new_df = pd.DataFrame(list(zip(labels, data)), columns=["labels", "values"])

                    # now do tukey
                    MultiComp = MultiComparison(new_df["values"], new_df["labels"])
                    result = MultiComp.tukeyhsd()
                    new_tukey_df = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])
                    new_tukey_df.to_csv("../acoustic_similarity_tukey/tukey_results/" + speaker + "_" + combo1 + "_" + combo2 + "_tukey_three.csv")

                    # find which ones are true
                    for index, row in new_tukey_df.iterrows():
                        a = None
                        b = None
                        if row['reject']:
                            if row['group1'] == 0:
                                a = combo1 + combo2
                            elif row['group1'] == 1:
                                a = combo1
                            else:
                                a = combo2
                            if row['group2'] == 0:
                                b = combo1 + combo2
                            elif row['group2'] == 1:
                                b = combo1
                            else:
                                b = combo2
                            comparison_three_results.append(a + " different than " + b)

    # total_two = 0
    # for second_emotion in emotions:
    #     comparison_three.append(second_emotion + "_" + emotion)
    
    # for i in range(len(comparison_three)):
    #     for j in range(len(comparison_three)):
    #         for k in range(len(comparison_three)):
    #             if i != j and i != k and j!= k:
    #                 group1 = [comparison_three[i] + "/" + comparison_three[j], comparison_three[j] + "/" + comparison_three[i]]
    #                 group2 = [comparison_three[i] + "/" + comparison_three[k], comparison_three[k] + "/" + comparison_three[i]]

    #                 possible_combos = []
    #                 for group in group1:
    #                     for group_2 in group2:
    #                         possible_combos.append([group, group_2])

    #                 comparison = None
    #                 combo1 = None
    #                 combo2 = None

    #                 df = None
    #                 for combo in possible_combos:
    #                     df = tukey_file[(tukey_file["group1"] == combo[0]) & (tukey_file["group2"] == combo[1])]
    #                     if df.empty:
    #                         df = tukey_file[(tukey_file["group1"] == combo[1]) & (tukey_file["group2"] == combo[0])]
    #                         if not df.empty:
    #                             combo1 = combo[1]
    #                             combo2 = combo[0]
    #                             break
    #                     else:
    #                         combo1 = combo[0]
    #                         combo2 = combo[1]
    #                         break

    #                 if df.shape[0] > 1:
    #                     df = df.iloc[0]

    #                 if combo1.split("/")[0] == combo2.split("/")[0]:
    #                     comparison = combo1.split("/")[0]
    #                     combo1 = combo1.split("/")[1]
    #                     combo2 = combo2.split("/")[1]
    #                 else:
    #                     comparison = combo1.split("/")[1]
    #                     combo1 = combo1.split("/")[0]
    #                     combo2 = combo2.split("/")[0]

    #                 is_failed = False
    #                 for fails in failed:
    #                     if fails[0] == comparison and fails[1] == combo1 and fails[2] == combo2 or fails[0] == comparison and fails[1] == combo2 and fails[2] == combo1:
    #                         is_failed = True

    #                 if list(df['reject'])[0] and not is_failed:
    #                     comparison_three_results += 1

    #                 total_two += 1
    # comparison_three_percent = comparison_three_results/total_two

    return comparison_two_results, comparison_three_results

if __name__=="__main__":
    failed_tukey = {}

    neutral_question = open("../acoustic_similarity_tukey/neutral_question.csv", "w")
    neutral_question_writer = csv.writer(neutral_question, delimiter=',')
    neutral_question_writer.writerow(["speaker", "question1", "answer1", "question2", "answer2", "question3", "answer3"])
    anger_question = open("../acoustic_similarity_tukey/anger_question.csv", "w")
    anger_question_writer = csv.writer(anger_question, delimiter=',')
    anger_question_writer.writerow(["speaker", "question1", "answer1", "question2", "answer2", "question3", "answer3"])
    elation_question = open("../acoustic_similarity_tukey/elation_question.csv", "w")
    elation_question_writer = csv.writer(elation_question, delimiter=',')
    elation_question_writer.writerow(["speaker", "question1", "answer1", "question2", "answer2", "question3", "answer3"])
    sadness_question = open("../acoustic_similarity_tukey/sadness_question.csv", "w")
    sadness_question_writer = csv.writer(sadness_question, delimiter=',')
    sadness_question_writer.writerow(["speaker", "question1", "answer1", "question2", "answer2", "question3", "answer3"])
    panic_question = open("../acoustic_similarity_tukey/panic_question.csv", "w")
    panic_question_writer = csv.writer(panic_question, delimiter=',')
    panic_question_writer.writerow(["speaker", "question1", "answer1", "question2", "answer2", "question3", "answer3"])

    emotions = ["neutral", "anger", "elation", "panic", "sadness"]

    for initial_csv_file in glob.glob("../acoustic_similarity_tukey/*_tukey_sanity_check.csv"):
        csv_file = open(initial_csv_file)
        csv_file_reader = csv.reader(csv_file, delimiter=',')

        speaker = str(initial_csv_file.split('/')[-1].split('_')[0])

        line_count = 0
        for row in csv_file_reader:
            if line_count == 0:
                line_count += 1
            else:
                # check that none of the neutral_anger/anger_anger and neutral_anger/neutral_neutral things fail tukey
                if row[1].split("/")[0] == row[2].split("/")[0]:
                    var1 = row[1].split("/")[0].split("_")[0]
                    var2 = row[1].split("/")[0].split("_")[1]
                    compare_var_1 = var1 + "_" + var1
                    compare_var_2 = var2 + "_" + var2

                    compare = None
                    if row[1].split("/")[1] == compare_var_1:
                        compare = compare_var_2
                    elif row[1].split("/")[1] == compare_var_2:
                        compare = compare_var_1

                    if compare:
                        if row[2].split("/")[1] == compare:
                            if (row[7] == "False"):
                                if speaker not in failed_tukey:
                                    failed_tukey[speaker] = [(row[1].split("/")[0], compare_var_1, compare_var_2)]
                                else:
                                    failed_tukey[speaker].append((row[1].split("/")[0], compare_var_1, compare_var_2))

                    # Is neutral_anger different from neutral_neutral, neutral_elation, neutral_sadness, neutral_panic
                    # neutral_anger/neutral_neutral vs. neutral_anger/neutral_elation
                    if row[7] == "False":
                        if row[1].split("/")[1] != row[2].split("/")[1]:
                            if speaker not in failed_tukey:
                                failed_tukey[speaker] = [(row[1].split("/")[0], row[1].split("/")[1], row[2].split("/")[1])]
                            else:
                                failed_tukey[speaker].append((row[1].split("/")[0], row[1].split("/")[1], row[2].split("/")[1]))
                    
                # Is neutral_anger different from anger_anger, elation_anger, sadness_anger, panic_anger
                elif row[1].split("/")[1] == row[2].split("/")[1]:
                    if row[7] == "False":
                        if row[1].split("/")[1] != row[2].split("/")[1]:
                            if speaker not in failed_tukey:
                                failed_tukey[speaker] = [(row[1].split("/")[0], row[1].split("/")[1], row[2].split("/")[1])]
                            else:
                                failed_tukey[speaker].append((row[1].split("/")[0], row[1].split("/")[1], row[2].split("/")[1]))

        neutral_one = None
        anger_one = None
        elation_one = None
        sadness_one = None
        panic_one = None
        # question 1
        for csv_file1 in glob.glob("../acoustic_similarity_tukey/*_anova.csv"):
            new_speaker = str(csv_file1.split('/')[-1].split('_')[0])

            if new_speaker == speaker:
                open_file = open(csv_file1)
                neutral_one = emotion_comparison("neutral", emotions, open_file, failed_tukey[speaker])
                open_file = open(csv_file1)
                anger_one = emotion_comparison("anger", emotions, open_file, failed_tukey[speaker])
                open_file = open(csv_file1)
                elation_one = emotion_comparison("elation", emotions, open_file, failed_tukey[speaker])
                open_file = open(csv_file1)
                sadness_one = emotion_comparison("sadness", emotions, open_file, failed_tukey[speaker])
                open_file = open(csv_file1)
                panic_one = emotion_comparison("panic", emotions, open_file, failed_tukey[speaker])

                for csv_file3 in glob.glob("../acoustic_similarity/*_full.csv"):
                    new_speaker2 = str(csv_file1.split('/')[-1].split('_')[0])
                    if new_speaker2 == new_speaker:
                        # question 2, 3
                        csv_file2 = open(initial_csv_file)
                        open_file = open(csv_file3)
                        neutral_two, neutral_three = emotion_comparison_tukey("neutral", emotions, csv_file2, failed_tukey[speaker], open_file, speaker)
                        csv_file2 = open(initial_csv_file)
                        open_file = open(csv_file3)
                        anger_two, anger_three = emotion_comparison_tukey("anger", emotions, csv_file2, failed_tukey[speaker], open_file, speaker)
                        csv_file2 = open(initial_csv_file)
                        open_file = open(csv_file3)
                        elation_two, elation_three = emotion_comparison_tukey("elation", emotions, csv_file2, failed_tukey[speaker], open_file, speaker)
                        csv_file2 = open(initial_csv_file)
                        open_file = open(csv_file3)
                        sadness_two, sadness_three = emotion_comparison_tukey("sadness", emotions, csv_file2, failed_tukey[speaker], open_file, speaker)
                        csv_file2 = open(initial_csv_file)
                        open_file = open(csv_file3)
                        panic_two, panic_three = emotion_comparison_tukey("panic", emotions, csv_file2, failed_tukey[speaker], open_file, speaker)

                        # write
                        neutral_question_writer.writerow([speaker, "neutral", neutral_one, "neutral", neutral_two, "neutral", neutral_three])
                        anger_question_writer.writerow([speaker, "anger", anger_one, "anger", anger_two, "anger", anger_three])
                        elation_question_writer.writerow([speaker, "elation", elation_one, "elation", elation_two, "elation", elation_three])
                        sadness_question_writer.writerow([speaker, "sadness", sadness_one, "sadness", sadness_two, "sadness", sadness_three])
                        panic_question_writer.writerow([speaker, "panic", panic_one, "panic", panic_two, "panic", panic_three])