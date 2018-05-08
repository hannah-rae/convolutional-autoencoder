import numpy as np
from glob import glob
from sklearn.naive_bayes import GaussianNB

results_dw = glob('/home/hannah/src/MastcamCAE/results/DW_mse_udr_12-8-3_7-5-3_nodrop_epochs15/*')
results_train = glob('/home/hannah/src/MastcamCAE/results/train_mse_udr_12-8-3_7-5-3_nodrop_epochs15/*')

# Convert filenames to a list of integers (MSE)
results_dw = [[int(a.split('/')[-1].split('_')[0])] for a in results_dw]
results_train = [[int(a.split('/')[-1].split('_')[0])] for a in results_train]

# Create positive and negative labels
labels_dw = [1]*len(results_dw)
labels_train = [0]*len(results_train)

# Shuffle the lists with same seed as other models
np.random.seed(42)
np.random.shuffle(results_dw)
np.random.shuffle(results_train)

# Convert to training and eval sets
train_data = np.concatenate([results_train[:98700], results_dw[:300]])
train_labels = np.concatenate([labels_train[:98700], labels_dw[:300]])
weights = np.add(1, np.multiply(train_labels, 328))

eval_data = np.concatenate([results_train[98700:], results_dw[300:]])
eval_labels = np.concatenate([labels_train[98700:], labels_dw[300:]])

clf = GaussianNB()
clf.fit(train_data, train_labels, sample_weight=weights)

acc = clf.score(eval_data, eval_labels)
print acc

tn = 0
tp = 0
fp = 0
fn = 0
for i, p in enumerate(clf.predict_proba(eval_data)):
    p_t, p_n = p
    if p_t >= 0.5 and eval_labels[i] == 0:
        tn += 1
    elif p_t >= 0.5 and eval_labels[i] == 1:
        fn += 1
    elif p_t < 0.5 and eval_labels[i] == 0:
        fp += 1
    else:
        tp += 1

print "False positives", fp
print "False negative", fn
print "True positives", tp
print "True negatives", tn