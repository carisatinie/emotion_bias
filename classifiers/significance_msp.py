import numpy as np
from matplotlib import pyplot as plt

'''
MSP: Graphs for Gender bias
Uses Tukey's features (highest performing feature set for MSP)
Only uses emotions that have significant difference between male and female.
'''

# MSP -- Tukey
# neutral, sad, anger

# male neutral 80.20231214	10.54913295	0	0.5780346821	8.670520231
# sad 80.85106383	14.89361702	0	2.127659574	2.127659574
# anger 69.93318486	17.59465479	0	0.2227171492	12.24944321

# f neutral 84.68624064	10.47783535	0.4029936672	0.2302820956	4.202648244
# sad 82.12290503	11.17318436	0.5586592179	0	6.145251397
# anger 80.10362694	13.78238342	0.207253886	0.518134715	5.388601036

# predicted as
m_neutral = np.array([80.20231214, 80.85106383, 69.93318486])
m_happiness = np.array([10.54913295, 14.89361702, 17.59465479])
m_sadness = np.array([0, 0, 0])
m_surprise = np.array([0.5780346821, 2.127659574, 0.2227171492])
m_anger = np.array([8.670520231, 2.127659574, 12.24944321])

f_neutral = np.array([84.68624064, 82.12290503, 80.10362694])
f_happiness = np.array([10.47783535, 11.17318436, 13.78238342])
f_sadness = np.array([0.4029936672, 0.5586592179, 0.207253886])
f_surprise = np.array([0.2302820956, 0, 0.518134715])
f_anger = np.array([4.202648244, 6.145251397, 5.388601036])

barWidth = 0.3
r1 = np.arange(3)
r2 = [x + barWidth for x in r1]

plt.bar(r1, m_neutral, color='#DE1414', width=barWidth, edgecolor='white', label='neutral')
plt.bar(r1, m_happiness, color='#DEC614', width=barWidth, edgecolor='white', label='happiness', bottom=m_neutral)
plt.bar(r1, m_sadness, color='#67DE14', width=barWidth, edgecolor='white', label='sadness', bottom=m_neutral+m_happiness)
plt.bar(r1, m_surprise, color='#1491DE', width=barWidth, edgecolor='white', label='surprise', bottom=m_neutral+m_happiness+m_sadness)
plt.bar(r1, m_anger, color='#8814DE', width=barWidth, edgecolor='white', label='anger', bottom=m_neutral+m_happiness+m_sadness+m_surprise)

plt.bar(r2, f_neutral, color='#DE1414', width=barWidth, edgecolor='white')
plt.bar(r2, f_happiness, color='#DEC614', width=barWidth, edgecolor='white', bottom=f_neutral)
plt.bar(r2, f_sadness, color='#67DE14', width=barWidth, edgecolor='white', bottom=f_neutral+f_happiness)
plt.bar(r2, f_surprise, color='#1491DE', width=barWidth, edgecolor='white', bottom=f_neutral+f_happiness+f_sadness)
plt.bar(r2, f_anger, color='#8814DE', width=barWidth, edgecolor='white', bottom=f_neutral+f_happiness+f_sadness+f_surprise)

plt.xlabel('Emotions')
plt.ylabel('Prediction Accuracies')
plt.title('MSP: Tukey, Significant Differences between Male/Female')
plt.xticks([r + barWidth/2 for r in range(3)], ['neutral', 'sadness', 'anger'])
plt.legend()
plt.show()