from glob import glob
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Get the reconstruction error values as a list of integers
results = glob('/Users/hannahrae/src/autoencoder/MastcamCAE/results/test_set_reducedtrain/*')
scores = [int(a.split('/')[-1].split('_')[0]) for a in results]
scores.sort()

gaps = []
for i in range(len(scores)):
    # If this is the last value in the list, we are done
    if i+1 == len(scores):
        continue
    gaps.append(abs(scores[i+1]-scores[i])) # don't need to take abs value since monotonic increase

plt.plot(gaps)
plt.title('Difference in reconstruction error across test examples')
plt.ylabel('Difference in error of adjacent examples, sorted by ascending error')
plt.xlabel('Position in list of examples, sorted by ascending error')
plt.show()