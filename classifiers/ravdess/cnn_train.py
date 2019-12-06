import torch
import torch.utils.data as data
import csv
import argparse
import torch.nn as nn
import numpy as np
import librosa
import os
import glob
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import librosa.display
import collections
import itertools

batch_size = 20
mean = 0

def spectrogram_extractor(input_folder, input_csv):
    global mean
    # i think there's something wrong with the encoder

    csv_open = open(input_csv, "r")
    csv_reader = csv.reader(csv_open, delimiter=',')

    file_to_label = {}
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            file_to_label[row[0][36:]] = float(row[-1])

    labels = []

    count = 0
    spectrograms = []
    for wave_file in glob.glob(input_folder + "/*.wav"):
        count += 1
        file_name = wave_file.split("/")[-1]
        # load
        y, sr = librosa.load(input_folder + file_name)
        # MFCCs
        # mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length = hop_length)
        # delta features
        # mfcc_delta = librosa.feature.delta(mfcc)
        # stack
        # stacked_mfcc = np.vstack([mfcc, mfcc_delta])
        spectrogram = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000))

        # clip out the most frequent ones
        most_common = collections.Counter(i for i in list(itertools.chain.from_iterable(spectrogram))).most_common(1)[0][0]
        new_spectrogram = []
        longest_length = 0
        for i in range(len(spectrogram)):
            intermediate = []
            for j in range(len(spectrogram[0])):
                if spectrogram[i][j] != most_common:
                    intermediate.append(spectrogram[i][j])
            new_spectrogram.append(intermediate)
            if len(intermediate) > longest_length:
                longest_length = len(intermediate)

        for i in range(len(new_spectrogram)):
            if len(new_spectrogram[i]) < longest_length:
                new_spectrogram[i] = np.pad(np.array(new_spectrogram[i]), (0, longest_length - len(new_spectrogram[i])), 'constant')

        new_spectrogram = np.array(new_spectrogram)

        # normalize data since they're all pretty small (also maybe try after?)
        # y = (x - min)/(max - min)
        # try standardization
        min_value = np.min(new_spectrogram)
        max_value = np.max(new_spectrogram)
        mean_value = np.mean(new_spectrogram)
        std = np.std(new_spectrogram)

        for i in range(len(new_spectrogram)):
            for j in range(len(new_spectrogram[i])):
                new_spectrogram[i][j] = (new_spectrogram[i][j] - min_value)/(max_value - min_value)
                # new_spectrogram[i][j] = (new_spectrogram[i][j] - mean_value)/std

        stacked_mfcc = new_spectrogram[np.newaxis, np.newaxis, :, :]
        # print(stacked_mfcc.shape)
        # sys.exit()
        # if count == 2:
        #     plt.subplot(4, 2, 1)
        #     librosa.display.specshow(spectrogram)
        #     plt.colorbar(format='%+2.0f dB')
        #     plt.title('spectrogram')
        #     plt.show()

        #     sys.exit()
        # print(stacked_mfcc.shape)

        # append
        spectrograms.append(stacked_mfcc)
        labels.append(file_to_label[file_name] - 1)

    print('finished')

    # sys.exit()

    # pad
    # first find the longest sequence
    max_len_3 = 0
    # max_len_2 = 0
    # max_len_1 = 0
    for spectrogram in spectrograms:
        if spectrogram.shape[3] > max_len_3:
            max_len_3 = spectrogram.shape[3]
        # if spectrogram.shape[2] > max_len_2:
        #     max_len_2 = spectrogram.shape[2]
        # if spectrogram.shape[1] > max_len_1:
        #     max_len_1 = spectrogram.shape[1]

    augmented_spectrograms = []

    # then pad the rest
    new_count = 0
    padding = []
    for spectrogram in spectrograms:
        new_count += 1
        pad_3 = 0
        # pad_2 = 0
        # pad_1 = 0
        if spectrogram.shape[3] < max_len_3:
            pad_3 = max_len_3 - spectrogram.shape[3]
            # print('pad', pad_3)
            # print('max length', max_len_3)
        # if spectrogram.shape[2] < max_len_2:
        #     pad_2 = max_len_2 - spectrogram.shape[2]
        # if spectrogram.shape[1] < max_len_1:
        #     pad_1 = max_len_1 - spectrogram.shape[1]
        # if pad_3 or pad_2 or pad_1:
        if pad_3:
            padding.append(pad_3)
            intermediate = []
            for i in range(len(spectrogram)):
                # print('spectrogram', spectrogram[i])
                if pad_3 % 2 == 0:
                    intermediate.append(np.pad(spectrogram[i], ((0, 0), (0, 0), (pad_3//2, pad_3//2)), 'constant', constant_values=(0, 0)))
                else:
                    intermediate.append(np.pad(spectrogram[i], ((0, 0), (0, 0), (pad_3//2, pad_3//2 + 1)), 'constant', constant_values=(0, 0)))
            augmented_spectrograms.append(intermediate)

        else:
            augmented_spectrograms.append(spectrogram)

    # truncating by mean of padding
    # print('counter', collections.Counter(padding))
    # print('mean', np.mean(padding))
    # new_mean = int(np.mean(padding))
    new_mean = int(np.max(padding))

    further_augmented_spectrograms = []
    for spectrogram in augmented_spectrograms:
        if mean == 0:
            start = new_mean//2
            end = -(new_mean//2)
            mean = new_mean
        else:
            start = mean//2
            end = -(mean//2)
        # print(start)
        # print(end)
        # print('len', len(spectrogram[0][0][start:end]))
        further_augmented_spectrograms.append([spectrogram[0][0][start:end]])

    augmented_spectrograms = further_augmented_spectrograms
    # print(augmented_spectrograms)

        # print('counter', collections.Counter(i for i in list(itertools.chain.from_iterable(augmented_spectrograms[new_count - 1][0][0]))).most_common(1))
        # print('whole size', augmented_spectrograms[0][0][0].shape[0]*augmented_spectrograms[0][0][0].shape[1])
        # if new_count != 1:
        #     dist = np.linalg.norm(np.array(augmented_spectrograms[new_count-1]) - np.array(augmented_spectrograms[new_count-2]))
        #     print('distance', dist)

    # FOR SURE INPUT IS THE PROBLEM

    features = torch.FloatTensor(augmented_spectrograms)
    # subtract 1 from every label
    labels = torch.FloatTensor(labels)
    # print('features shape', features.shape)
    # print('labels shape', labels.shape)
    data_input = data.TensorDataset(features, labels)
    data_loader = data.DataLoader(data_input, batch_size=batch_size, shuffle=True)
    

    return data_loader

def loader(csv_file):
    csv_open = open(csv_file, "r")
    csv_reader = csv.reader(csv_open, delimiter=',')

    # label is the last row
    features = []
    labels = []

    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            features.append(list(map(float, row[1:-1])))
            labels.append(list(map(float, row[-1])))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    data_input = data.TensorDataset(features, labels)
    data_loader = data.DataLoader(data_input, batch_size=batch_size, shuffle=True)

    return data_loader

class Attention(nn.Module):
    # Bahdanau (MLP) attention
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.query = nn.Parameter(torch.FloatTensor(hidden_size, 1))
    
    def forward(self, output):
        # multiply query to output
        alignment_score = torch.matmul(output, self.query) # 50, 57, 100; query is 100, 1 = 50, 57, 1

        # print('alignment scores shape', alignment_score.shape)

        # softmax alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_score.view(output.shape[0], -1), dim=1) # weights of each step, 50, 57

        # print('attention weights shape', attn_weights.shape)
        # print('output shape', output.shape)

        # weighted average (element wise), weighted time over attention score
        context_vector = attn_weights.unsqueeze(2) * output # 50, 57 * 50, 57, 100 = 50, 57, 100

        # print('context vector shape', context_vector.shape)

        # sum over time
        sum_over_time = torch.sum(context_vector, dim=1)

        return sum_over_time # 50, 100 (same as last_hidden[0])

class cnn_model(nn.Module):
    def __init__(self):
        # hyperparameters
        self.epochs = 4
        self.classes = 8
        self.learning_rate = 0.1

        super(cnn_model, self).__init__()
        # first set of convolutions
        self.layer_one = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=1, stride=1))
        # what dimension is the cnn working on? -> this should be fine
        # self.layer_one = nn.Conv2d(1, 16, kernel_size=20, stride=5, padding=1)
        # self.layer_three = nn.ReLU()
        # self.layer_four = nn.MaxPool2d(kernel_size=1, stride=1)

        # second set of convolutions
        self.layer_two = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        # dropout layer
        self.dropout = nn.Dropout(p=0.1)
        # fully connected layers

        # put an lstm here so it's not 58k, figure out the dimensions (one of the features is about time) and put the lstm along the time sequence
        # can use the last step of the LSTM because it's the summary; look at the numbers (print shape, there should be a 32, and 200 ish - 200 would be the time)
        # self.lstm = nn.LSTM(1024, 100, batch_first=True) # 50, 1024, 57 where 57 is time, 1024 is embedding, 50 is batch size
        # self.lstm = nn.LSTM(704, 100, batch_first=True)
        self.lstm = nn.LSTM(124, 50, batch_first=True) #124

        # attention layer - for time dimension
        # self.attention = Attention(100)
        self.attention = Attention(50)

        # linear layers
        # self.fclayer_one = nn.Linear(100, 25)
        self.fclayer_one = nn.Linear(5984, 500) #3840
        self.fclayer_two = nn.Linear(500, self.classes)

    def forward(self, x):
        # print('HERE')
        # x = torch.FloatTensor(np.squeeze(x, axis=1).shape)
        x = torch.FloatTensor(x)
        # print(x.shape)
        # print('x', x)
        # print(x.type())

        # we will put in a random tensor with 10, 1, 128, 95
        # y = np.random.random_sample((10, 1, 128, 95))
        # print(y.dtype)
        # y = torch.from_numpy(y)
        # y = torch.Tensor(y)
        # print(y.type())

        # print('x', x)
        # out_one = self.layer_one(x)
        # print('after cnn 1', out_one.shape)
        # out_one_2 = self.layer_three(out_one)
        # print('after relu', out_one_2)
        # out_one_3 = self.layer_four(out_one_2)
        # print('after pool', out_one_3.shape)
        # sys.exit()

        # 0s out with standardization; gets same value for everything (prob just bias) with normalization

        # for i in range(out_one.shape[1]):
            # print(out_one[0][0][i])
            # print('counter', collections.Counter(i for i in list(itertools.chain.from_iterable(out_one[0][i]))).most_common(1))
            # print('whole size', augmented_spectrograms[0][0][0].shape[0]*augmented_spectrograms[0][0][0].shape[1])
        # sys.exit()
        # out_two = self.layer_two(out_one)
        # print('after cnn 2', out_two.shape)
        # out_three = out_two.reshape(out_two.shape[0], out_two.shape[1]*out_two.shape[2], out_two.shape[3])
        # out_three = out_one_3.reshape(out_one_3.shape[0], out_one_3.shape[1]*out_one_3.shape[2], out_one_3.shape[3])
        # print('after reshape', out_three)
        out_three = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        # out_four = self.dropout(out_three)
        # print('after dropout', out_four.shape)
        # out_five = out_four.transpose(1, 2)
        # print('after dropout', out_four)
        # hidden_vector, last_hidden = self.lstm(out_three)
        # print('after lstm', hidden_vector.shape)
        # # print('after lstm', last_hidden[0][-1])
        # # add attention
        # sum_over_time = self.attention(hidden_vector) # last_hidden[0][-1] use this step for just linear

        # print('after attention', sum_over_time.shape)
        # print('after attention', sum_over_time)

        out_six = self.fclayer_one(out_three)
        # print('after linear 1', out_six.shape)
        out_seven = self.fclayer_two(out_six)
        # print('after linear 2', out_seven)
        return out_seven

def train(model, criterion, optimizer, loader):
    losses = []
    accuracies = []
    print('training')
    for epoch in range(model.epochs):
        for i, (features, labels) in enumerate(loader):
            labels = labels.type(torch.LongTensor)
            output = model(features)
            current_loss = criterion(output, labels)
            losses.append(current_loss.item())

            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()

            number = labels.size(0)
            one, predicted = torch.max(output.data, 1)
            correct = (predicted == labels).sum().item()
            accuracies.append(correct/number)
            if (i + 1) % 5 == 0:
                # print('first', one)
                # print('predicted', predicted)
                # print('labels', labels)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, model.epochs, i + 1, len(loader), current_loss.item(),
                          (correct / number) * 100))

def validate(model, loader):
    # evaluate model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in loader:
            # print(features)
            output = model(features)
            # print('output shape', output.shape)
            # print('output shape', output.data.shape)
            # print('output', output)
            one, predicted = torch.max(output, 1)
            # print(_)
            # print(predicted)
            # print('shape tensor2', _.shape)
            # print('shape long tensor2', predicted.shape)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print('values', one)
            print('predicted val', predicted)
            print('labels', labels)

        print('Validation accuracy: {} %'.format((correct/total)*100))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainfolder", required=True)
    parser.add_argument("--validationfolder", required=True)
    parser.add_argument("--traincsv", required=True)
    parser.add_argument("--validationcsv", required=True)
    parsed_args = parser.parse_args()

    # load the training data
    training_loader = spectrogram_extractor(parsed_args.trainfolder, parsed_args.traincsv)
    # print('HERE')
    # load the validation data
    validation_loader = spectrogram_extractor(parsed_args.validationfolder, parsed_args.validationcsv)
    # print('HERE1')
    model = cnn_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    # print('HERE2')
    # train
    train(model, criterion, optimizer, training_loader)

    # validate
    validate(model, validation_loader)

    # save the model
    torch.save(model.state_dict(), "cnn_lstm_attention_model.pt")
