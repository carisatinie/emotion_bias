import numpy as np
from matplotlib import pyplot as plt

# NEUTRAL
# rav: rf_m_neutral
# msp: corr_neutral, mfcc_neutral, acoustic neutral, ma_neutral, rf neutral
# masc: acoustic, ma, rf

barWidth = 0.3
r1 = np.arange(9)
r2 = [x + barWidth for x in r1]

# rav_rf_neutral_m = [0, 100, 0, 0, 0]
# rav_rf_neutral_f = [0, 0, 0, 33.33, 66.66]
# msp_corr_neutral_m = [90, 5.6, 0.1, 0, 4.3] # 90.02890173	5.563583815	0.07225433526	0	4.335260116
# msp_corr_neutral_f = [88.2, 6.5, 1.2, 0, 4.1] # 88.1980426	6.5054692	1.208981002	0	4.087507196
# msp_mfcc_neutral_m = [82, 9.3, 0, 0.3, 7.9] # 82.44219653	9.320809249	0	0.289017341	7.947976879
# msp_mfcc_neutral_f = [89, 8.2, 0.46, 0.06, 2] # 89.34945308	8.175014393	0.4605641911	0.05757052389	1.957397812
# msp_a_neutral_m = [70, 16, 0.8, 2.2, 11.2] # 69.50867052	16.32947977	0.7947976879	2.167630058	11.19942197
# msp_a_neutral_f = [74, 12, 1.8, 1.4, 11] # 73.69027058	12.03223949	1.842256765	1.439263097	10.99597006
# msp_ma_neutral_m = [80, 11, 0, 0.6, 8.7] # 80.20231214	10.54913295	0	0.5780346821	8.670520231
# msp_ma_neutral_f = [85, 10, 0.4, 0.2, 4.2] # 84.68624064	10.47783535	0.4029936672	0.2302820956	4.202648244
# msp_rf_neutral_m = [74, 13, 0.6, 1.5, 11] # 74.49421965	12.78901734	0.5780346821	1.51734104	10.62138728
# msp_rf_neutral_f = [96, 3.5, 0, 0, 0.2] # 96.25791595	3.511801957	0	0	0.2302820956
# masc_a_neutral_m = [56, 11, 16, 10, 8] # 56.10972569	10.72319202	15.71072319	9.725685786	7.855361596
# masc_a_neutral_f = [51, 5.3, 27, 12.6, 4] # 51.2195122	5.284552846	26.82926829	12.60162602	4.06504065
# masc_ma_neutral_m = [61, 7, 19, 6.1, 6.2] # 61.22194514	6.982543641	19.45137157	6.109725686	6.234413965
# masc_ma_neutral_f = [66, 7, 19.5, 3.3, 4.1] # 65.85365854	7.317073171	19.51219512	3.25203252	4.06504065
# masc_rf_neutral_m = [41, 14, 16.6, 18, 9.9] # 41.27182045	13.840399	16.58354115	18.45386534	9.850374065
# masc_rf_neutral_f = [49, 10, 25.6, 2.4, 13] # 48.7804878	9.756097561	25.6097561	2.43902439	13.41463415

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
