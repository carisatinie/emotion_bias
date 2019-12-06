import numpy as np
from matplotlib import pyplot as plt

# RAVDESS -- MFCC
# surprise

# male 6.25	31.25	0	37.5	25

# female 5.263157895	5.263157895	15.78947368	68.42105263	5.263157895

# predicted as
m_neutral = np.array([6.25])
m_happiness = np.array([31.25])
m_sadness = np.array([0])
m_surprise = np.array([37.5])
m_anger = np.array([25])

f_neutral = np.array([5.263157895])
f_happiness = np.array([5.263157895])
f_sadness = np.array([15.78947368])
f_surprise = np.array([68.42105263])
f_anger = np.array([5.263157895])

barWidth = 0.1
r1 = np.arange(1)
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

plt.xlim(-0.1,0.2)
plt.xlabel('Emotions')
plt.ylabel('Prediction Accuracies')
plt.title('RAVDESS: MFCC, Significant Differences between Male/Female')
plt.xticks([r + barWidth/2 for r in range(1)], ['surprise'])
plt.legend()
plt.show()