from matplotlib import pyplot as plt 
import numpy as np

''' RAVDESS Graphs '''
objects = ('Tukey', 'Acoustic', 'MFCC', 'MFCC+Ac', 'RF')
x_pos = np.arange(len(objects))

# Male
f1_male = [0.5757, 0.3227, 0.2975, 0.3899, 0.2569]
plt.bar(x_pos, f1_male, align='center', alpha=0.5)
plt.xticks(x_pos, objects)
plt.ylabel("F1 score")
plt.title("RAVDESS Male F1 Scores on Different Feature Sets")
plt.ylim((0, 0.6)) 
for i, v in enumerate(f1_male):
    plt.text(x_pos[i] - 0.25, v, str(v))
plt.show()

# Female
f1_female = [0.3154, 0.3182, 0.4764, 0.4278, 0.3269]
plt.bar(x_pos, f1_female, align='center', color=(0.6, 0, 1, 0.5))
plt.xticks(x_pos, objects)
plt.ylabel("F1 score")
plt.title("RAVDESS Female F1 Scores on Different Feature Sets")
plt.ylim((0, 0.6)) 
for i, v in enumerate(f1_female):
    plt.text(x_pos[i] - 0.25, v, str(v))
plt.show()

# Both
f1_both = [0.2776, 0.2811, 0.3756, 0.4905, 0.2949]
plt.bar(x_pos, f1_both, align='center', color=(0, 0.5, 0))
plt.xticks(x_pos, objects)
plt.ylabel("F1 score")
plt.title("RAVDESS Both F1 Scores on Different Feature Sets")
plt.ylim((0, 0.6)) 
for i, v in enumerate(f1_both):
    plt.text(x_pos[i] - 0.25, v, str(v))
plt.show()