import pandas as pd
import sys
import ast
from operator import or_
from functools import reduce

# diff 4 = textual emotion definitely influences how it's said

emotions = ["neutral", "anger", "elation", "panic", "sadness"]
speakers = {}
for emotion in emotions:
    for i in range(len(emotions)):
        speakers[emotion + "_" + emotions[i]] = 0

for emotion in emotions:
    open_file = open("../acoustic_similarity_tukey/" + emotion + "_question.csv")
    df = pd.read_csv(open_file)

    for i in range(1, 69):
        current_speaker = {}
        for emotion2 in emotions:
            current_speaker[emotion + "_" + emotion2] = 0

        answer_df = df[df["speaker"]==i]["answer2"]
        final_set = ()
        for j in range(1, len(answer_df.index)):
            if j == 1:
                final_set = reduce(or_, [set(ast.literal_eval(answer_df.iloc[0])), set(ast.literal_eval(answer_df.iloc[1]))])
            else:
                final_set = reduce(or_, [final_set, set(ast.literal_eval(answer_df.iloc[j]))])
        
        final_values = []
        for value in final_set:
            comparison = value.split(' ')[0]
            count = 0
            for second_value in final_set:
                if comparison in second_value:
                    count += 1
            if count > 1:
                if comparison not in final_values:
                    final_values.append(comparison)

        for value in final_values:
            for key in current_speaker.keys():
                if key in value:
                    current_speaker[key] += 1

        for key in current_speaker.keys():
            if current_speaker[key] >= 4:
                speakers[key] += 1

for key in speakers.keys():
    speakers[key] = speakers[key]/68

# print(speakers)

# {'neutral_neutral': 0.8382352941176471, 'neutral_anger': 0.9852941176470589, 'neutral_elation': 0.8823529411764706, 'neutral_panic': 0.8529411764705882, 'neutral_sadness': 0.8970588235294118, 'anger_neutral': 0.9558823529411765, 'anger_anger': 0.9852941176470589, 'anger_elation': 0.9705882352941176, 'anger_panic': 0.9705882352941176, 'anger_sadness': 0.9852941176470589, 'elation_neutral': 0.9852941176470589, 'elation_anger': 0.9705882352941176, 'elation_elation': 0.9705882352941176, 'elation_panic': 0.9852941176470589, 'elation_sadness': 0.9852941176470589, 'panic_neutral': 1.0, 'panic_anger': 1.0, 'panic_elation': 1.0, 'panic_panic': 0.9411764705882353, 'panic_sadness': 0.9411764705882353, 'sadness_neutral': 0.7058823529411765, 'sadness_anger': 0.8235294117647058, 'sadness_elation': 0.9264705882352942, 'sadness_panic': 0.7352941176470589, 'sadness_sadness': 0.7352941176470589}

average = {"neutral": 0, "anger": 0, "elation": 0, "panic": 0, "sadness": 0}
for key in speakers.keys():
    for key2 in average.keys():
        if key.split("_")[0] == key2:
            average[key2] += speakers[key]

for key in average.keys():
    average[key] = average[key]/5

print(average)
# {'neutral': 0.8911764705882353, 'anger': 0.973529411764706, 'elation': 0.9794117647058824, 'panic': 0.9764705882352942, 'sadness': 0.7852941176470589}