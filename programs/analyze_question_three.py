import pandas as pd
import sys
import ast
from operator import or_
from functools import reduce

# diff 1 = textual emotion definitely influences how it's said

emotions = ["neutral", "anger", "elation", "panic", "sadness"]
speakers = {}
for emotion in emotions:
    for i in range(len(emotions)):
        speakers[emotions[i] + "_" + emotion] = 0

for emotion in emotions:
    open_file = open("../acoustic_similarity_tukey/" + emotion + "_question.csv")
    df = pd.read_csv(open_file)

    for i in range(1, 69):
        current_speaker = {}
        for emotion2 in emotions:
            current_speaker[emotion2 + "_" + emotion] = 0

        answer_df = df[df["speaker"]==i]["answer3"]
        final_set = ()
        for j in range(1, len(answer_df.index)):
            if j == 1:
                final_set = reduce(or_, [set(ast.literal_eval(answer_df.iloc[0])), set(ast.literal_eval(answer_df.iloc[1]))])
            else:
                final_set = reduce(or_, [final_set, set(ast.literal_eval(answer_df.iloc[j]))])
        
        final_values = []
        for value in final_set:
            comparison = value.split(' ')[0]
            if comparison.count("_") == 2:
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

print(speakers)

# {'neutral_neutral': 1.0, 'anger_neutral': 1.0, 'elation_neutral': 1.0, 'panic_neutral': 1.0, 'sadness_neutral': 1.0, 'neutral_anger': 1.0, 'anger_anger': 1.0, 'elation_anger': 1.0, 'panic_anger': 1.0, 'sadness_anger': 1.0, 'neutral_elation': 1.0, 'anger_elation':1.0, 'elation_elation': 1.0, 'panic_elation': 1.0, 'sadness_elation': 1.0, 'neutral_panic': 1.0, 'anger_panic': 1.0, 'elation_panic': 1.0, 'panic_panic': 1.0, 'sadness_panic': 1.0, 'neutral_sadness': 1.0, 'anger_sadness': 1.0, 'elation_sadness': 1.0, 'panic_sadness': 1.0, 'sadness_sadness': 1.0}

average = {"neutral": 0, "anger": 0, "elation": 0, "panic": 0, "sadness": 0}
for key in speakers.keys():
    for key2 in average.keys():
        if key.split("_")[0] == key2:
            average[key2] += speakers[key]

for key in average.keys():
    average[key] = average[key]/5

print(average)
# {'neutral': 1.0, 'anger': 1.0, 'elation': 1.0, 'panic': 1.0, 'sadness': 1.0}

# for count >= 3
# {'neutral_neutral': 1.0, 'anger_neutral': 0.9852941176470589, 'elation_neutral': 1.0, 'panic_neutral': 1.0, 'sadness_neutral': 1.0,'neutral_anger': 1.0, 'anger_anger': 0.9852941176470589, 'elation_anger': 0.9852941176470589, 'panic_anger': 1.0, 'sadness_anger': 0.9852941176470589, 'neutral_elation': 1.0, 'anger_elation': 1.0, 'elation_elation': 1.0, 'panic_elation': 1.0, 'sadness_elation': 1.0, 'neutral_panic': 1.0, 'anger_panic': 1.0, 'elation_panic': 1.0, 'panic_panic': 1.0, 'sadness_panic': 1.0, 'neutral_sadness': 1.0, 'anger_sadness': 0.9852941176470589, 'elation_sadness': 1.0, 'panic_sadness': 1.0, 'sadness_sadness': 1.0}
# {'neutral': 1.0, 'anger': 0.9911764705882353, 'elation': 0.9970588235294118, 'panic': 1.0, 'sadness': 0.9970588235294118}

# for count >= 4
# {'neutral_neutral': 0.9852941176470589, 'anger_neutral': 0.9264705882352942, 'elation_neutral': 0.9411764705882353, 'panic_neutral': 0.9558823529411765, 'sadness_neutral': 0.9705882352941176, 'neutral_anger': 1.0, 'anger_anger': 0.9558823529411765, 'elation_anger': 0.9558823529411765, 'panic_anger': 0.9411764705882353, 'sadness_anger': 0.9558823529411765, 'neutral_elation': 1.0, 'anger_elation': 0.9411764705882353, 'elation_elation': 0.9558823529411765, 'panic_elation': 0.9852941176470589, 'sadness_elation': 1.0, 'neutral_panic': 0.9852941176470589, 'anger_panic': 0.9558823529411765, 'elation_panic': 0.9558823529411765, 'panic_panic': 0.9117647058823529, 'sadness_panic': 0.9558823529411765, 'neutral_sadness': 0.9852941176470589, 'anger_sadness': 0.9558823529411765, 'elation_sadness': 0.9558823529411765, 'panic_sadness': 0.9558823529411765, 'sadness_sadness': 0.9852941176470589}
# {'neutral': 0.9911764705882353, 'anger': 0.9470588235294117, 'elation': 0.9529411764705882, 'panic': 0.95, 'sadness': 0.973529411764706}