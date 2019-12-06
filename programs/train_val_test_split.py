import csv
import pandas as pd
import numpy as np

all_data = pd.read_csv("../final_data/labeled_features_ravdess.csv")

speech = all_data[all_data.speech_song==1]
# song = all_data[all_data.speech_song==2]

train_speech, validate_speech, test_speech = np.split(speech.sample(frac=1), [int(.8*len(speech)), int(.9*len(speech))])
# train_song, validate_song, test_song = np.split(song.sample(frac=1), [int(.8*len(song)), int(.9*len(song))])

train_speech.to_csv("../final_data/ravdess/speech_train_data.csv", index=False)
validate_speech.to_csv("../final_data/ravdess/speech_val_data.csv", index=False)
test_speech.to_csv("../final_data/ravdess/speech_test_data.csv", index=False)

# train_song.to_csv("final_data/ravdess/song_train_data.csv", index=False)
# validate_song.to_csv("final_data/ravdess/song_val_data.csv", index=False)
# test_song.to_csv("final_data/ravdess/song_test_data.csv", index=False)
