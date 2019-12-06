from matplotlib import pyplot as plt 
import numpy as np


# corr_m_happy	[ 1 11 0 11 10]	corr_f_happy	[1 5 3 2 7]
# mfcc_m_surprised	[1 5 0 6 4]	mfcc_f_surprised	[ 1 1 3 13 1]
# rf_m_neutral	[0 4 0 0 0]	rf_f_neutral	[0 0 0 1 2]
# rf_m_surprised	[0 8 0 8 0]	rf_f_surprised	[ 0 3 0 14 2]

barWidth = 0.25
r1 = np.arange(5)
r2 = [x + barWidth for x in r1]

# corr happy: 
c_m_accuracies = [3.0303, 33.3333, 0, 33.3333, 3.0303]
c_f_accuracies = [5.5556, 27.7778, 16.6667, 11.1111, 38.8889]

# Make the plot
plt.bar(r1, c_m_accuracies, color='#8EA4C6', width=barWidth, edgecolor='white', label='male')
plt.bar(r2, c_f_accuracies, color='#D0A8D3', width=barWidth, edgecolor='white', label='female')

plt.xlabel('Actual Emotion')
plt.ylabel('Prediction Accuracy')
plt.xticks([r + barWidth/2 for r in range(len(c_m_accuracies))], ['neutral', 'happiness', 'sadness', 'surprise', 'anger'])
plt.legend()
for i, v in enumerate(c_m_accuracies):
    plt.text(r1[i]-barWidth/2, v, str(int(v))+"%")
for i, v in enumerate(c_f_accuracies):
    plt.text(r1[i]+barWidth/2, v, str(int(v))+"%")
plt.show()