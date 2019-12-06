from matplotlib import pyplot as plt 
import numpy as np

''' RAVDESS Graphs '''
objects = ('Tukey', 'Acoustic', 'MFCC', 'MFCC+Ac', 'RF')
x_pos = np.arange(len(objects))

# Male
f1_male = [0.4838, 0.2221, 0.2010, 0.2073, 0.2106]
plt.bar(x_pos, f1_male, align='center', alpha=0.5)
plt.xticks(x_pos, objects)
plt.ylabel("F1 score")
plt.title("MSP Male F1 Scores on Different Feature Sets")
plt.ylim((0, 0.55)) 
for i, v in enumerate(f1_male):
    plt.text(x_pos[i] - 0.25, v, str(v))
plt.show()

# Female
f1_female = [0.5081, 0.2044, 0.1663, 0.1795, 0.1981]
plt.bar(x_pos, f1_female, align='center', color=(0.6, 0, 1, 0.5))
plt.xticks(x_pos, objects)
plt.ylabel("F1 score")
plt.title("MSP Female F1 Scores on Different Feature Sets")
plt.ylim((0, 0.55)) 
for i, v in enumerate(f1_female):
    plt.text(x_pos[i] - 0.25, v, str(v))
plt.show()

# Both
f1_both = [0.2213, 0.2136, 0.2158, 0.2209, 0.2141]
plt.bar(x_pos, f1_both, align='center', color=(0, 0.5, 0))
plt.xticks(x_pos, objects)
plt.ylabel("F1 score")
plt.title("MSP Both F1 Scores on Different Feature Sets")
plt.ylim((0, 0.55)) 
for i, v in enumerate(f1_both):
    plt.text(x_pos[i] - 0.25, v, str(v))
plt.show()