import glob
import csv
import pandas as pd
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
import scipy.stats as ss
import numpy as np
import sys

def anova_comparison(first_df, second_df, comparison, first_i, second_i):
    f_stat, p_value = ss.f_oneway(list(first_df["similarity"]), list(second_df["similarity"]))

    mean_first = np.mean(np.array(list(first_df["similarity"])))
    mean_second = np.mean(np.array(list(second_df["similarity"])))

    words = None
    if (mean_first - mean_second) < 0:
        words = first_i + " is closer to " + comparison +  " than " + second_i
    else:
        words = second_i + " is closer to " + comparison +  " than " + first_i

    return [first_i, second_i, comparison, f_stat, p_value, (mean_first - mean_second), words]

if __name__=="__main__":
    # all possible pairs of emotions
    emotions = ["neutral", "anger", "elation", "panic", "sadness"]
    comparisons = []
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            if i != j:
                comparisons.append(emotions[i] + "_" + emotions[j])

    # loops through each csv
    for initial_csv_file in glob.glob("../acoustic_similarity/*_full.csv"):
        speaker = initial_csv_file.split('/')[-1][:3]
        csv_file = open(initial_csv_file)
        pd_reader = pd.read_csv(csv_file)

        first_comparison = open("../acoustic_similarity_tukey/" + speaker + "_anova.csv", "w")
        first_comparison_writer = csv.writer(first_comparison, delimiter=',')
        first_comparison_writer.writerow(['pair_1', 'pair_2', 'comparing_var', 'f_stat', 'p_value', 'comparison', 'words'])

        # compare everything with everything
        frames = []
        all_vals = pd_reader.adjs.unique()
        for val in all_vals:
            frames.append(pd_reader[pd_reader["adjs"]==val][["adjs", "similarity"]])
        concat = pd.concat(frames)

        # 0. as sanity check, we can run Tukey on all of these comparisons to make sure no groups are rlly similar
        MultiComp = MultiComparison(concat["similarity"], concat["adjs"])
        result = MultiComp.tukeyhsd()
        tukey_df = pd.DataFrame(data=result._results_table.data[1:], columns=result._results_table.data[0])
        
        for comparison in comparisons:
            first_emotion = comparison.split('_')[0]
            second_emotion = comparison.split('_')[1]

            # Tukey test on neutral_neutral, anger_anger, neutral_elation, neutral_panic, neutral_sadness
            # read all rows for neutral_neutral
            first_i = first_emotion + "_" + first_emotion
            df1 = pd_reader[pd_reader["adjs"]==(comparison + "/" + first_i)][["adjs", "similarity"]]
            df1["similarity"] = df1["similarity"].astype(float)

            second_i = second_emotion + "_" + second_emotion
            df2 = pd_reader[pd_reader["adjs"]==(comparison + "/" + second_i)][["adjs", "similarity"]]
            df2["similarity"] = df2["similarity"].astype(float)

            leftover_emotion = emotions.copy()
            leftover_emotion.remove(first_emotion)
            leftover_emotion.remove(second_emotion)

            third_i = first_emotion + "_" + leftover_emotion[0]
            fourth_i = first_emotion + "_" + leftover_emotion[1]
            fifth_i = first_emotion + "_" + leftover_emotion[2]

            df3 = pd_reader[pd_reader["adjs"]==(comparison + "/" + third_i)][["adjs", "similarity"]] 
            if df3.empty:
                df3 = pd_reader[pd_reader["adjs"]==(third_i + "/" + comparison)][["adjs", "similarity"]] 
            df4 = pd_reader[pd_reader["adjs"]==(comparison + "/" + fourth_i)][["adjs", "similarity"]]
            if df4.empty:
                df4 = pd_reader[pd_reader["adjs"]==(fourth_i + "/" + comparison)][["adjs", "similarity"]]
            df5 = pd_reader[pd_reader["adjs"]==(comparison + "/" + fifth_i)][["adjs", "similarity"]]
            if df5.empty:
                df5 = pd_reader[pd_reader["adjs"]==(fifth_i + "/" + comparison)][["adjs", "similarity"]]
            
            df3["similarity"] = df3["similarity"].astype(float)
            df4["similarity"] = df4["similarity"].astype(float)
            df5["similarity"] = df5["similarity"].astype(float)

            # 1. is it more similar to neutral_neutral or anger_anger (one way ANOVA on the two above)
            output = anova_comparison(df1, df2, comparison, first_i, second_i)
            first_comparison_writer.writerow(output)

            # 2. is the similarity closer to neutral_neutral and anger_anger than neutral_elation, neutral_panic, neutral_sadness (i + other 3 emotions)
            #   (compare neutral_neutral and neutral_elation) - one way ANOVA
            #   (compare anger_anger and neutral_elation) - one way ANOVA
            output = anova_comparison(df1, df3, comparison, first_i, third_i)
            first_comparison_writer.writerow(output)
            output = anova_comparison(df2, df3, comparison, second_i, third_i)
            first_comparison_writer.writerow(output)
            
            output = anova_comparison(df1, df4, comparison, first_i, fourth_i)
            first_comparison_writer.writerow(output)
            output = anova_comparison(df2, df4, comparison, second_i, fourth_i)
            first_comparison_writer.writerow(output)

            output = anova_comparison(df1, df5, comparison, first_i, fifth_i)
            first_comparison_writer.writerow(output)
            output = anova_comparison(df2, df5, comparison, second_i, fifth_i)
            first_comparison_writer.writerow(output)

        first_comparison.close()

        tukey_df.to_csv("../acoustic_similarity_tukey/" + speaker + "_tukey_sanity_check.csv")