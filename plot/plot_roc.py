#print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy import interp

def compute_accuracy(data, labels, thresholds):
    max_acc = 0
    max_t = 0
    max_tn = 0
    max_tp = 0
    for t in thresholds:
        num_correct = 0
        tp = 0
        tn = 0
        for x, y in zip(data, labels):
            if x >= t and y == 1:
                num_correct += 1
                tp += 1
            elif x < t and y == 0:
                num_correct += 1
                tn += 1
        acc = float(num_correct) / 130.
        print 'threshold %f' % t
        print 'tp %d' % tp
        print 'tn %d' % tn
        if acc > max_acc:
            max_acc = acc
            max_t = t
            max_tp = tp
            max_tn = tn
    return max_acc, max_t, max_tp, max_tn

results = dict()
labels = dict()

# Import the data for CNN
results['cnn'] = np.ndarray([132])
for idx, line in enumerate(file('/Users/hannahrae/Documents/Grad School/NIPS2018/roc/cnn/cnn_probs.txt')):
    results['cnn'][idx] = float(line.rstrip().split()[1])

labels['cnn'] = np.ndarray([132])
for idx, line in enumerate(file('/Users/hannahrae/Documents/Grad School/NIPS2018/roc/cnn/cnn_true_classes.txt')):
    labels['cnn'][idx] = float(line.rstrip())

print labels['cnn']
print results['cnn']

# Import the data for Gaussian Naive Bayes
results['gnb'] = np.ndarray([132])
for idx, line in enumerate(file('/Users/hannahrae/Documents/Grad School/NIPS2018/roc/gnb/gnb_probs.txt')):
    results['gnb'][idx] = line.rstrip().split()[1]

labels['gnb'] = np.ndarray([132])
for idx, line in enumerate(file('/Users/hannahrae/Documents/Grad School/NIPS2018/roc/gnb/gnb_true_classes.txt')):
    labels['gnb'][idx] = line.rstrip()

# Import the data for Inception-V3 (long)
results['inc_long'] = np.ndarray([132])
labels['inc_long'] = np.ndarray([132])
for idx, line in enumerate(file('/Users/hannahrae/Documents/Grad School/NIPS2018/roc/inception_long/results.txt')):
    results['inc_long'][idx] = line.rstrip().split()[0]
    labels['inc_long'][idx] = line.rstrip().split()[1]

# Import the data for Inception-V3 (short)
results['inc_short'] = np.ndarray([132])
labels['inc_short'] = np.ndarray([132])
for idx, line in enumerate(file('/Users/hannahrae/Documents/Grad School/NIPS2018/roc/inception_short/results.txt')):
    results['inc_short'][idx] = line.rstrip().split()[0]
    labels['inc_short'][idx] = line.rstrip().split()[1]

# Import the data for Feedforward network
results['ffnn'] = np.ndarray([132])
for idx, line in enumerate(file('/Users/hannahrae/Documents/Grad School/NIPS2018/roc/ffnn/ffnn_probs.txt')):
    results['ffnn'][idx] = line.rstrip().split()[1]

labels['ffnn'] = np.ndarray([132])
for idx, line in enumerate(file('/Users/hannahrae/Documents/Grad School/NIPS2018/roc/ffnn/ffnn_true_classes.txt')):
    labels['ffnn'][idx] = line.rstrip()

# Binarize the output
# y_labels = label_binarize(y_labels, classes=[0, 1])
# print y_labels.shape
# n_classes = y_labels.shape[1]
# print n_classes
# Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
#                                                     random_state=0)

# Learn to predict each class against the other
# classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
#                                  random_state=random_state))
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

fpr["cnn"], tpr["cnn"], _ = roc_curve(y_true=labels["cnn"], y_score=results["cnn"])
best_acc, threshold, tp, tn = compute_accuracy(results["cnn"], labels["cnn"], _)
print "CNN best accuracy %f threshold %f tp %d tn %d" % (best_acc, threshold, tp, tn)
roc_auc["cnn"] = auc(fpr["cnn"], tpr["cnn"])

fpr["gnb"], tpr["gnb"], _ = roc_curve(y_true=labels["gnb"], y_score=results["gnb"])
best_acc, threshold, tp, tn = compute_accuracy(results["gnb"], labels["gnb"], _)
print "GNB best accuracy %f threshold %f tp %d tn %d" % (best_acc, threshold, tp, tn)
roc_auc["gnb"] = auc(fpr["gnb"], tpr["gnb"])

fpr["inc_long"], tpr["inc_long"], _ = roc_curve(y_true=labels["inc_long"], y_score=results["inc_long"])
best_acc, threshold, tp, tn = compute_accuracy(results["inc_long"], labels["inc_long"], _)
print "Inc long best accuracy %f threshold %f tp %d tn %d" % (best_acc, threshold, tp, tn)
roc_auc["inc_long"] = auc(fpr["inc_long"], tpr["inc_long"])

fpr["inc_short"], tpr["inc_short"], _ = roc_curve(y_true=labels["inc_short"], y_score=results["inc_short"])
best_acc, threshold, tp, tn = compute_accuracy(results["inc_short"], labels["inc_short"], _)
print "Inc short best accuracy %f threshold %f tp %d tn %d" % (best_acc, threshold, tp, tn)
roc_auc["inc_short"] = auc(fpr["inc_short"], tpr["inc_short"])

fpr["ffnn"], tpr["ffnn"], _ = roc_curve(y_true=labels["ffnn"], y_score=results["ffnn"])
best_acc, threshold, tp, tn = compute_accuracy(results["ffnn"], labels["ffnn"], _)
print "FFNN best accuracy %f threshold %f tp %d tn %d" % (best_acc, threshold, tp, tn)
roc_auc["ffnn"] = auc(fpr["ffnn"], tpr["ffnn"])

# Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_true=y_labels, y_score=y_cnn, pos_label=1)
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print roc_auc_score(y_true=labels["cnn"], y_score=results["cnn"])
print roc_auc_score(y_true=labels["gnb"], y_score=results["gnb"])
print roc_auc_score(y_true=labels["inc_long"], y_score=results["inc_long"])
print roc_auc_score(y_true=labels["inc_short"], y_score=results["inc_short"])
print roc_auc_score(y_true=labels["ffnn"], y_score=results["ffnn"])

plt.figure()
lw = 2
plt.plot(fpr["cnn"], tpr["cnn"], color='black',
         lw=4, label='SAMMIE (AUC = %0.2f)' % roc_auc["cnn"])

plt.plot(fpr["inc_short"], tpr["inc_short"], color='black', linestyle='-',
         lw=lw, label='Inception-V3 (short) (AUC = %0.2f)' % roc_auc["inc_short"])

plt.plot(fpr["inc_long"], tpr["inc_long"], color='black', linestyle='--',
         lw=lw, label='Inception-V3 (long) (AUC = %0.2f)' % roc_auc["inc_long"])

plt.plot(fpr["ffnn"], tpr["ffnn"], color='black', linestyle=':',
         lw=lw, label='FFNN (AUC = %0.2f)' % roc_auc["ffnn"])

plt.plot(fpr["gnb"], tpr["gnb"], color='black', linestyle='-.',
         lw=lw, label='Gaussian Naive Bayes (AUC = %0.2f)' % roc_auc["gnb"])

plt.plot([0, 1], [0, 1], color='lightgray', lw=lw, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for tested classifiers')
plt.legend(loc="lower right")
plt.show()