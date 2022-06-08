# Class for testing the implementation of the SVM for cones
import pandas as pd
import sklearn.metrics as sk
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import params
from classSVM import classSVM

from small_SVM import Tree

data = pd.read_csv("../data/cod-rna.csv", sep=";")
data = data.to_numpy()

x = data[:, 1::]
y = data[:, 0]

# as data set is big, use only a partition for learning
kf = KFold(n_splits=25, shuffle=True, random_state=0)
test_index, train_index = list(kf.split(x))[0]

x_test = x[test_index]
y_test = y[test_index]

x = x[train_index]
y = y[train_index]

# do cross validation
kf = KFold(n_splits=6, shuffle=True, random_state=0)

acc_cone = []
prec_cone = []
rec_cone = []
f1_cone = []
acc_coef = []
prec_coef = []
rec_coef = []
f1_coef = []

for train_index, val_index in kf.split(x):
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # using the cone-svm for training and testing
    SVM_cone = Tree(x_train, y_train)
    classif_cone = classSVM(SVM_cone, x_val)

    acc_cone.append(sk.accuracy_score(y_val, classif_cone))
    prec_cone.append(sk.precision_score(y_val, classif_cone))
    rec_cone.append(sk.recall_score(y_val, classif_cone))
    f1_cone.append(sk.f1_score(y_val, classif_cone))

    # calculate classical SVM-result
    SVM = SVC(kernel='rbf', gamma=params.KERNEL_PARAM_SVM, C=params.C_SVM)

    SVM.fit(x_train, y_train)
    classification = SVM.predict(x_val)

    acc_coef.append(sk.accuracy_score(y_val, classification))
    prec_coef.append(sk.precision_score(y_val, classification))
    rec_coef.append(sk.recall_score(y_val, classification))
    f1_coef.append(sk.f1_score(y_val, classification))

# Prediction for test set
SVM_cone = Tree(x, y)
classif_cone = classSVM(SVM_cone, x_test)

# calculate classical SVM-result
SVM = SVC(kernel='rbf', gamma=params.KERNEL_PARAM_SVM, C=params.C_SVM)

SVM.fit(x, y)
classification = SVM.predict(x_test)

print("acc_svm_cone", acc_cone, sk.accuracy_score(y_test, classif_cone))
print("prec_svm_cone", prec_cone, sk.precision_score(y_test, classif_cone))
print("rec_SVM_cone", rec_cone, sk.recall_score(y_test, classif_cone))
print("f1_cone", f1_cone, sk.f1_score(y_test, classif_cone))
print("###############################################")
print("acc_svm", acc_coef, sk.accuracy_score(y_test, classification))
print("prec_svm", prec_coef, sk.precision_score(y_test, classification))
print("rec_SVM", rec_coef, sk.recall_score(y_test, classification))
print("f1", f1_coef, sk.f1_score(y_test, classification))
