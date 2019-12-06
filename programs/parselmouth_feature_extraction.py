import glob
import numpy as np
import pandas as pd
import parselmouth 
from parselmouth.praat import call
import argparse
import csv
import ast
import os

#https://github.com/drfeinberg/PraatScripts/blob/master/Measure%20Pitch%2C%20HNR%2C%20Jitter%2C%20Shimmer%2C%20and%20Formants.ipynb

# This is the function to measure source acoustics using default male parameters.

# we will extract all features here, including porting over speaking rate and rating

def measureFeatures(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    intensityMin = None
    intensityMax = None
    intensityMean = None
    # try:
    intensity = call(sound, "To Intensity", 100.0, 0.0, False)
    intensityMean = intensity.get_average()

    sorted_intensity = np.sort(intensity.values)
    # remove the negative values that show up for some reason
    augmented_sorted_intensity = np.array([intensity for intensity in sorted_intensity[0] if intensity >= 0])
    intensityMin = augmented_sorted_intensity[0]
    intensityMax = augmented_sorted_intensity[-1]
    # except:
    #     intensityMin = 100000000
    #     intensityMax = 100000000
    #     intensityMean = 100000000
    
    # minf0 = None
    # maxf0 = None
    # meanF0 = None
    # try:
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch

    sorted_pitch = []
    for i in range(len(pitch.selected_array)):
        if pitch.selected_array[i][0] > 0:
            sorted_pitch.append(pitch.selected_array[i][0])
    print(pitch.selected_array)
    sorted_pitch = sorted(sorted_pitch)
    minf0 = sorted_pitch[0]
    maxf0 = sorted_pitch[-1]
    # except:
    #     minf0 = 100000000
    #     maxf0 = 100000000
    #     meanF0 = 100000000

    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    # localJitter = 100000000
    # localShimmer = 100000000
    # try:
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    # except:
    #     pass
    
    
    return duration, intensityMin, intensityMax, intensityMean, minf0, maxf0, meanF0, hnr, localJitter, localShimmer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folderpath", default="", required=True)
    parser.add_argument("--minpitch", default=75)
    parser.add_argument("--maxpitch", default=400)
    parser.add_argument("--outputfile", required=True)
    parser.add_argument("--mergefile", required=True)

    parsed_args = parser.parse_args()

    # create lists to put the results
    file_list = []
    duration_list = []
    intensity_min_list = []
    intensity_max_list = []
    intensity_mean_list = []
    F0_min_list = []
    F0_max_list = []
    mean_F0_list = []
    hnr_list = []
    local_jitter_list = []
    local_shimmer_list = []

    redo_list = []
    line_count = 0
    with open('masc_redos.txt') as txt_file:
        redo_list = txt_file.read().splitlines()

    count = 0
    # Go through all the wave files in the folder and measure all the acoustics
    for r, d, f in os.walk(parsed_args.folderpath):
        for folder in d:
            for wave_file in glob.glob(os.path.join(r, folder) + "/*.wav"):
                file_name = os.path.join(r, folder) + "/" + wave_file.split("/")[-1]

                if file_name not in redo_list:
                    pass
                else:
                    print(count)
                    count += 1
                    sound = parselmouth.Sound(wave_file)
                    (duration, intensityMin, intensityMax, intensityMean, f0min, f0max, meanF0, hnr, localJitter, localShimmer) = measureFeatures(
                        sound, parsed_args.minpitch, parsed_args.maxpitch, "Hertz")

                    file_list.append(file_name) # make an ID list
                    duration_list.append(duration) # make duration list
                    intensity_min_list.append(intensityMin)
                    intensity_max_list.append(intensityMax)
                    intensity_mean_list.append(intensityMean)
                    F0_min_list.append(f0min)
                    F0_max_list.append(f0max)
                    mean_F0_list.append(meanF0) # make a mean F0 list
                    hnr_list.append(hnr) #add HNR data
                    
                    # add raw jitter and shimmer measures
                    local_jitter_list.append(localJitter)
                    local_shimmer_list.append(localShimmer)

    # Add the data to Pandas
    df = None
    stacked_columns = np.column_stack([file_list, intensity_min_list, intensity_max_list,
                                        intensity_mean_list, F0_min_list, F0_max_list, mean_F0_list, hnr_list, 
                                        local_jitter_list, local_shimmer_list])

    columns = ['name', 'min_intensity', 'max_intensity', 'mean_intensity', 'min_pitch', 'max_pitch', 'mean_pitch', 'HNR', 
                'local_jitter', 'local_shimmer']

    df = pd.DataFrame(stacked_columns, columns=columns)

    # reload the data so it's all numbers
    # other_data = pd.read_csv(parsed_args.mergefile)
    # merged = df.merge(other_data, on='name')

    # merged.to_csv(parsed_args.outputfile, index=False)
    df.to_csv(parsed_args.outputfile, index=False)
