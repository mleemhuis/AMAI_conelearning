# clustering the elements in groups of positives and negatives
# and learning a SVM for each group
import numpy as np
import params
from SVM import SVM_class
import numpy.ma as ma

class Tree:
    def __init__(self, x, y, in_el=np.inf):
        self.x = x
        self.y = y

        # calculate the children

        # calculate SVM
        self.SVM = SVM_class(kerneltype=params.KERNEL_TYPE, kernelparameter=params.KERNEL_PARAM, c=params.C)

        self.SVM.fit(x, y)
        classification = self.SVM.predict_plane(x)

        # Update hyperplane by using only correct classified elements
        correct_cl = ~np.bool_(np.abs(classification - y))
        self.SVM.fit(x[correct_cl, :], y[correct_cl])

        classification = self.SVM.predict_plane(x)

        # split the set into the positive and the negative classified elements
        masked_classification = ma.masked_less(classification, 0)

        neg_el_mask = masked_classification.mask
        x_left = x[neg_el_mask, :]
        y_left = y[neg_el_mask]

        # classification error
        err = y_left - classification[neg_el_mask]
        left_error = np.size(err[err != 0])

        pos_el_mask = ~masked_classification.mask
        x_right = x[pos_el_mask, :]
        y_right = y[pos_el_mask]

        err = y_right - classification[pos_el_mask]
        right_error = np.size(err[err != 0])

        # calculate incorrect positives, negatives
        if in_el - left_error < params.TR or left_error < params.TR:
            self.left_child = -1
        else:
            self.left_child = Tree(x_left, y_left, left_error)
        if in_el - right_error < params.TR or right_error < params.TR:
            self.right_child = 1
        else:
            self.right_child = Tree(x_right, y_right, right_error)

    def classify(self, x_new):
        if np.size(np.shape(x_new), 0) == 1:
            x_new = np.reshape(x_new, (1, x_new.size))

        cl = self.SVM.predict_plane(x_new)
        if cl == -1:
            if self.left_child == -1:
                return -1
            else:
                return self.left_child.classify(x_new)
        elif cl == 1:
            if self.right_child == 1:
                return 1
            else:
                return self.right_child.classify(x_new)