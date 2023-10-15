import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

row_labels = [120, 180, 240, 300, 360, 420]
column_labels = [14, 18, 22, 26, 30]

for file, mini, maxi in [('1/ig.csv', 0.0, 0.3), ('1/pv.csv', 0.0, 1.0), ('1/ra.csv', 0.825, 1.0),
                         ('5/ig.csv', 0.0, 0.3), ('5/pv.csv', 0.0, 1.0), ('5/ra.csv', 0.825, 1.0),
                         ('50/ig.csv', 0.0, 0.3), ('50/pv.csv', 0.0, 1.0), ('50/ra.csv', 0.825, 1.0)]:
    data = pd.read_csv(file, header=None)
    matrix = data.values

    sns.set()
    plt.figure(figsize=(3.8, 3.5))  # Adjust the figure size as needed

    # Customize the heatmap, including the colormap, annot, and other parameters
    heatmap = sns.heatmap(matrix, annot=False, fmt='.2f', cmap="crest", vmin=mini, vmax=maxi,
                          xticklabels=column_labels, yticklabels=row_labels)

    path = file.split('/')
    plt.savefig(path[0] + '/' + path[0] + '_' + path[1].split('.')[0] + '.png')
    plt.show()
