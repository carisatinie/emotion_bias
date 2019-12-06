import numpy as np
from matplotlib import pyplot as plt

# HAPPINESS
# rav: corr_m_happy
# msp: mfcc happy, acoustic, ma, rf
# masc: acoustic, ma, rf

barWidth = 0.3
r1 = np.arange(9)
r2 = [x + barWidth for x in r1]

m_rav_corr = 3.03030303	33.33333333	0	33.33333333	3.03030303
m_msp_mfcc
m_msp_a
m_msp_ma
m_msp_rf
m_mas_a
m_mas_ma
m_mas_rf

f_rav_happy = 5.555555556	27.77777778	16.66666667	11.11111111	38.88888889
f_msp_mfcc
f_msp_a
f_msp_ma
f_msp_rf
f_mas_a
f_mas_ma
f_mas_rf

# rav rf, msp corr, msp mfcc, msp a, msp ma, msp rf, masc a, masc ma, masc rf
# predicted emotions, actual emotion = neutral
male_neutral = np.array([0, 90.02890173, 82.44219653, 69.50867052, 80.20231214, 74.49421965, 56.10972569, 61.22194514, 41.27182045])
male_happiness = np.array([100, 5.563583815, 9.320809249, 16.32947977, 10.54913295, 12.78901734, 10.72319202, 6.982543641, 13.840399])
male_sadness = np.array([0, 0.07225433526, 0, .7947976879, 0, .5780346821, 15.71072319, 19.45137157, 16.58354115])
male_surprise = np.array([0, 0, 0.289017341, 2.167630058, 0.5780346821, 1.51734104, 9.725685786, 6.109725686, 18.45386534])
male_anger = np.array([0, 4.335260116, 7.947976879, 11.19942197, 8.670520231, 10.62138728, 7.855361596, 6.2234413965, 9.850374065])

female_neutral = np.array([0, 88.1980426, 89.34945308, 73.69027058, 84.68624064, 96.25791595, 51.2195122, 65.85365854, 48.7804878])
female_happiness = np.array([0, 6.5054692, 8.175014393, 12.03223949, 10.47783535, 3.511801957, 5.284552846, 7.317073171, 9.756097561])
female_sadness = np.array([0, 1.208981002, 0.4605641911, 1.842256765, .4029936672, 0, 26.82926829, 19.51219512, 25.6097561])
female_surprise = np.array([33.33, 0, 0.05757052389, 1.439263097, .2302820956, 0, 12.60162602, 3.25203252, 2.43902439])
female_anger = np.array([66.66, 4.087507196, 1.957397812, 10.99597006, 4.202648244, 0.2302820956, 4.06504065, 4.06504065, 13.41463415])

plt.bar(r1, male_neutral, color='#DE1414', width=barWidth, edgecolor='white', label='male')
plt.bar(r1, male_happiness, color='#DEC614', width=barWidth, edgecolor='white', label='male', bottom=male_neutral)
plt.bar(r1, male_sadness, color='#67DE14', width=barWidth, edgecolor='white', label='male', bottom=male_neutral+male_happiness)
plt.bar(r1, male_surprise, color='#1491DE', width=barWidth, edgecolor='white', label='male', bottom=male_neutral+male_happiness+male_sadness)
plt.bar(r1, male_anger, color='#8814DE', width=barWidth, edgecolor='white', label='male', bottom=male_neutral+male_happiness+male_sadness+male_surprise)

plt.bar(r2, female_neutral, color='#DE1414', width=barWidth, edgecolor='white', label='female')
plt.bar(r2, female_happiness, color='#DEC614', width=barWidth, edgecolor='white', label='female', bottom=female_neutral)
plt.bar(r2, female_sadness, color='#67DE14', width=barWidth, edgecolor='white', label='female', bottom=female_neutral+female_happiness)
plt.bar(r2, female_surprise, color='b', width=barWidth, edgecolor='white', label='female', bottom=female_neutral+female_happiness+female_sadness)
plt.bar(r2, female_anger, color='#8814DE', width=barWidth, edgecolor='white', label='female', bottom=female_neutral+female_happiness+female_sadness+female_surprise)

plt.show()
