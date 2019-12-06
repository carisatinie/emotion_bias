import matplotlib.pyplot as plt

plt.figure()
plt.ylim(0, 100)

labels = ["Anger", "Elation", "Panic"]
labels1 = ["Anger", "Elation", "Panic"]
percentages = [60.29, 85.29, 67.65]
percentages1 = [69.12, 82.35, 57.35]
plt.bar(labels, percentages, align='center', alpha=0.5)
plt.xticks(labels)
plt.axhline(y=50, linewidth=1, color='grey', linestyle='--')
plt.ylabel('% Speakers Supporting Hypothesis')
plt.title('Spoken Emotion Neutral')
plt.savefig("neutral_q1.png")

plt.figure()
plt.ylim(0, 100)
plt.bar(labels1, percentages1, align='center', alpha=0.5, color='#b19cd9')
plt.xticks(labels1)
plt.axhline(y=50, linewidth=1, color='grey', linestyle='--')
plt.ylabel('% Speakers Supporting Hypothesis')
plt.title('Spoken Emotion Sadness')
plt.savefig("sadness_q1.png")

plt.figure()
plt.ylim(0, 100)
labels2 = ["Neutral", "Anger", "Elation", "Panic", "Sadness"]
percentages2 = [89.12, 97.35, 97.94, 97.65, 78.53]
plt.bar(labels2, percentages2, align='center', alpha=0.5)
plt.xticks(labels2)
plt.axhline(y=50, linewidth=1, color='grey', linestyle='--')
plt.ylabel('Average % Speakers Differing Across Textual Emotion')
plt.savefig("q2.png")

plt.figure()
plt.ylim(0, 5.1)
percentages3 = [0, 0.88, 0.29, 0, 0.29]
plt.bar(labels2, percentages3, align='center', alpha=0.5)
plt.xticks(labels2)
plt.ylabel('Average % Speakers Similar Across Textual Emotion')
plt.title('3/4 Comparisons')
plt.savefig("34_q3.png")

plt.figure()
plt.ylim(0, 5.1)
percentages4 = [0.88, 5.29, 4.71, 5, 2.65]
plt.bar(labels2, percentages4, align='center', alpha=0.5, color='#b19cd9')
plt.xticks(labels2)
plt.ylabel('Average % Speakers Similar Across Textual Emotion')
plt.title('4/4 Comparisons')
plt.savefig("44_q3.png")