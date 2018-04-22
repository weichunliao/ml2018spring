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

# for q11
def digit2label(ys, label):
    label_y = []
    for y in ys:
        if y == label:
            label_y.append(+1)
        else:
            label_y.append(-1)
    return np.array(label_y)

y11 = digit2label(train_y, 0)

range_of_c = [-5, -3, -1, 1, 3]
Cs11 = [10**x for x in range_of_c]
w_norm_11 = []
for C in Cs11:
    clf = SVC(C=C, kernel='linear')
    clf.fit(train_X, y11)
    w = clf.coef_.reshape(-1)
    w_norm_11.append(np.sqrt(np.sum(w**2)))

    plt.figure(figsize=(6,4), dpi=110)

plt.title("Q11")
# plt.plot(range_of_c, w_norm_11)
plt.plot(w_norm_11)
plt.xlabel('C') 
plt.ylabel('||w||')
plt.xticks(range(len(Cs11)), Cs11)
plt.show()
for c, e in zip(Cs11, w_norm_11):
    print('C='+str(c)+': '+str(e))
    
    
