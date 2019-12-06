import pandas as pd
import sys
import ast

# what percentage of speakers have neutral_anger, neutral_elation,
# neutral_panic, neutral_sadness closer to neutral_neutral than x_x

emotions = ["neutral", "anger", "elation", "panic", "sadness"]
speakers = {}
speakers_failed = {}
for emotion in emotions:
    for i in range(len(emotions)):
        if emotion != emotions[i]:
            speakers[emotion + "_" + emotions[i]] = 0
            speakers_failed[emotion + "_" + emotions[i]] = 0

for emotion in emotions:
    emotion_file = open("../acoustic_similarity_tukey/" + emotion + "_question.csv")
    df = pd.read_csv(emotion_file)

    for i in range(1, 69):
        answer = ast.literal_eval(df[df["speaker"]==i]["answer1"].iloc[0])
        for whole in answer:
            if whole[0] == 'True' and whole[1] == 'NotFailed':
                speakers[whole[3]] += 1
            elif whole[0] == 'False' and whole[1] == 'NotFailed':
                speakers_failed[whole[3]] += 1

for key in speakers.keys():
    speakers[key] = speakers[key]/68

for key in speakers_failed.keys():
    speakers_failed[key] = speakers_failed[key]/68

print(speakers)
print(speakers_failed)

# {'neutral_anger': 0.6029411764705882, 'neutral_elation': 0.8529411764705882, 'neutral_panic': 0.6764705882352942, 'neutral_sadness': 0.10294117647058823, 'anger_neutral': 0.07352941176470588, 'anger_elation': 0.058823529411764705, 'anger_panic': 0.07352941176470588, 'anger_sadness': 0.17647058823529413, 'elation_neutral': 0.11764705882352941, 'elation_anger': 0.07352941176470588, 'elation_panic': 0.058823529411764705, 'elation_sadness': 0.23529411764705882, 'panic_neutral': 0.14705882352941177, 'panic_anger': 0.14705882352941177, 'panic_elation': 0.14705882352941177, 'panic_sadness': 0.2647058823529412, 'sadness_neutral': 0.058823529411764705, 'sadness_anger': 0.6911764705882353, 'sadness_elation': 0.8235294117647058, 'sadness_panic': 0.5735294117647058}
# {'neutral_anger': 0.0, 'neutral_elation': 0.0, 'neutral_panic': 0.0, 'neutral_sadness': 0.0, 'anger_neutral': 0.0, 'anger_elation':0.014705882352941176, 'anger_panic': 0.014705882352941176, 'anger_sadness': 0.0, 'elation_neutral': 0.014705882352941176, 'elation_anger': 0.014705882352941176, 'elation_panic': 0.029411764705882353, 'elation_sadness': 0.0, 'panic_neutral': 0.04411764705882353, 'panic_anger': 0.0, 'panic_elation': 0.0, 'panic_sadness': 0.0, 'sadness_neutral': 0.014705882352941176, 'sadness_anger': 0.0, 'sadness_elation': 0.0, 'sadness_panic': 0.0}