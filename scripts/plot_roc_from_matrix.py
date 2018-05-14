import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np
from itertools import cycle

from sklearn.metrics import roc_curve, auc

def load_janus_csv(filename):
    data = [] 
    titles = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for i,row in enumerate(reader):
            d = {}
            if i==0:
                for item in row:
                    titles.append(item)
            else:
                for j,item in enumerate(row):
                    d[titles[j]] = item

                data.append(d)
        #print titles
    return (data, titles)

def handle(matrix_file, ground_truth_file):

    (data, titles) = load_janus_csv(ground_truth_file)
    identity = {}
    for item in data:
        identity[item['TEMPLATE_ID']] = item['SUBJECT_ID']

    (data, titles) = load_janus_csv(matrix_file)
    labels = []
    scores = []
    for item in data:
        tid = item['FILENAME'].split('/')[-1].replace('.tmpl','')
        sid = identity[tid]
        for i in range(1, len(titles)):
            tid2 = titles[i].split('/')[-1].replace('.tmpl','')
            sid2 = identity[tid2]
            vm = item[titles[i]]
            scores.append(float(vm))
            if sid2==sid:
                labels.append(1)
            else:
                labels.append(-1)

    return (labels, scores)

plt.figure(figsize=(7, 7), dpi=150)
ground_truth_file = sys.argv[2]
matrix_file = sys.argv[1]
(labels, scores) = handle(matrix_file, ground_truth_file)
print zip(scores, labels)
fpr, tpr, _ = roc_curve(np.array(labels), np.array(scores))
roc_auc = auc(fpr, tpr)
lw = 2
plt.plot(fpr, tpr, 
                 lw=lw, label='multi (auc = %f)' % roc_auc)

ground_truth_file = sys.argv[4]
matrix_file = sys.argv[3]
(labels, scores) = handle(matrix_file, ground_truth_file)
print zip(scores, labels)
fpr, tpr, _ = roc_curve(np.array(labels), np.array(scores))
roc_auc = auc(fpr, tpr)
lw = 2
plt.plot(fpr, tpr, 
                 lw=lw, label='single (auc = %f)' % roc_auc)


#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('multi-image.png')
