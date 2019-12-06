import numpy as np
from matplotlib import pyplot as plt

# MASC -- Masc
# neutral, happiness, sadness, surprise, anger

# predicted as
m_neutral = np.array([61.22194514, 11.55717762, 25.48015365, 10.59782609, 7.362784471])
m_happiness = np.array([6.982543641, 43.06569343, 4.737516005, 16.71195652, 27.04149933])
m_sadness = np.array([19.45137157, 4.866180049, 56.46606914, 9.510869565, 2.543507363])
m_surprise = np.array([6.109725686, 16.42335766, 10.11523688, 46.19565217, 14.19009371])
m_anger = np.array([6.234413965, 24.08759124, 3.201024328, 16.98369565, 48.86211513])

f_neutral = np.array([65.85365854, 10.35856574, 33.06451613, 7.53968254, 13.06306306])
f_happiness = np.array([7.317073171, 45.41832669, 4.838709677, 21.82539683, 20.27027027])
f_sadness = np.array([19.51219512, 3.187250996, 52.01612903, 11.50793651, 5.855855856])
f_surprise = np.array([3.25203252, 15.53784861, 4.838709677, 46.42857143, 13.96396396])
f_anger = np.array([4.06504065, 25.49800797, 5.241935484, 12.6984127, 46.84684685])

barWidth = 0.3
r1 = np.arange(5)
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
plt.title('MASC: MFCC+Acoustic, Significant Differences between Male/Female')
plt.xticks([r + barWidth/2 for r in range(5)], ['neutral', 'happiness', 'sadness', 'surprise', 'anger'])
plt.legend()
plt.show()