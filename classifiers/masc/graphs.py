from matplotlib import pyplot as plt 
import numpy as np

''' RAVDESS Graphs '''
objects = ('Tukey', 'Acoustic', 'MFCC', 'MFCC+Ac', 'RF')
x_pos = np.arange(len(objects))

# Male
f1_male = [0.4308, 0.3701, .4948, .5109, .3548]
plt.bar(x_pos, f1_male, align='center', alpha=0.5)
plt.xticks(x_pos, objects)
plt.ylabel("F1 score")
plt.title("MASC Male F1 Scores on Different Feature Sets")
plt.ylim((0, 0.55)) 
for i, v in enumerate(f1_male):
    plt.text(x_pos[i] - 0.25, v, str(v))
plt.show()

# Female
f1_female = [.4436, .3592, .4884, .5111, .3448]
plt.bar(x_pos, f1_female, align='center', color=(0.6, 0, 1, 0.5))
plt.xticks(x_pos, objects)
plt.ylabel("F1 score")
plt.title("MASC Female F1 Scores on Different Feature Sets")
plt.ylim((0, 0.55)) 
for i, v in enumerate(f1_female):
    plt.text(x_pos[i] - 0.25, v, str(v))
plt.show()

# Both
f1_both = [.4307, .3668, .4855, .4943, .3397]
plt.bar(x_pos, f1_both, align='center', color=(0, 0.5, 0))
plt.xticks(x_pos, objects)
plt.ylabel("F1 score")
plt.title("MASC Both F1 Scores on Different Feature Sets")
plt.ylim((0, 0.55)) 
for i, v in enumerate(f1_both):
    plt.text(x_pos[i] - 0.25, v, str(v))
plt.show()