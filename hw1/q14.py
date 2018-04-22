import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def read_data(fname):
    X,y = [], []
    with open(fname,'r') as f:
        for line in f:
            d = line.split()
            X.append([float(a) for a in d[1:]])
            y.append(int(float(d[0])))
    return np.array(X), np.array(y)

train_X, train_y = read_data('./features.train')
test_X, test_y = read_data('./features.test')

def digit2label(ys, label):
    label_y = []
    for y in ys:
        if y == label:
            label_y.append(+1)
        else:
            label_y.append(-1)
    return np.array(label_y)

y14 = digit2label(train_y, 0)
Cs14 = [10**x for x in [-3, -2, -1, 0, 1]]


sv_dist_list14 = []

for C in Cs14:
    clf = SVC(kernel='rbf', C = C, gamma = 80)
    clf.fit(train_X, y14)
    alpha = np.abs(clf.dual_coef_.reshape(-1))
    sv_x = clf.support_vectors_
    sv_y = y14[clf.support_]
    w = 0
    for i in range(len(alpha)):
        for j in range(len(alpha)):
            w += alpha[i] * alpha[j] * sv_y[i] * sv_y[j] * \
                 np.exp(-80 * (np.sum((sv_x[i]-sv_x[j]) ** 2)))
    sv_dist_list14.append(1/np.sqrt(w))

plt.figure(figsize=(6,4), dpi=120)
plt.title("Q14") 
plt.plot(sv_dist_list14)
plt.xticks(range(len(Cs14)), Cs14)
plt.xlabel('C') 
plt.ylabel('distance')
plt.show()
for c, d in zip(Cs14, sv_dist_list14):
    print('C='+str(c)+': '+str(d))

