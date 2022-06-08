# creation of a SVM
# like a classical SVM but with b=0
# interpreted as dual and solved with quadratic programming
import cvxopt
from cvxopt import matrix
import numpy as np
import numpy.ma as ma
import params


class SVM_class:
    def __init__(self, kerneltype='gaussian', kernelparameter=0, c=np.inf, normalize=False):
        # possible options:
        # - kerneltype (rbf,...)
        # the error C (set to inf by default (meaning no error allowed))
        self.kerneltype = kerneltype
        self.kernelparameter = kernelparameter

        # check whether a cost matrix or only a single cost vector is defined.
        if np.size(c) == 1:
            self.c = [c, c]
        else:
            self.c = c

        self.normalize = normalize

        # initialize with null
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.n_alpha = None
        self.norms_x = None

    def fit(self, x, y):
        n = np.size(x, 0)
        # using quadratic programming with cvxopt
        # (see: http://cvxopt.org/userguide/coneprog.html#quadratic-programming)
        # standard definition of cvxopt:
        # max q^T x + (1/2)x^T P x
        # s.t. Gx <= h

        # translated to the SVM-approach:
        # min -[1...1]a^T (1/2) sum (a_i a_j y_i y_j K(x_i,x_j))
        # s.t. 0 <= a_i <= C
        # where the values of the vector a are searched for. a is 1xn
        # the cvxopt-matrix function is needed for initializing the arrays

        q = matrix(-1.0, (n, 1))

        P_arr = np.zeros((n, n))
        if self.normalize:
            # normalize data-points for training
            # calculate norms beforehand for reusability
            self.norms_x = np.zeros(n)
            for i in range(0, n):
                self.norms_x[i] = np.sqrt(self.kernel(x[i, :], x[i, :]))

            for i in range(0, n):
                for j in range(0, n):
                    P_arr[i, j] = y[i] * y[j] * self.kernel(x[i, :], x[j, :]) / (self.norms_x[i] * self.norms_x[j])
        else:
            for i in range(0, n):
                for j in range(0, n):
                    P_arr[i, j] = y[i] * y[j] * self.kernel(x[i, :], x[j, :])
        P = matrix(P_arr)

        # G is two times the identity matrix, one time for the >=0 and one time for <=C
        G = matrix(np.concatenate((-np.eye(n), np.eye(n)), 0))

        # the first half of h is zero, the second half c
        h = np.zeros((2 * n, 1))

        # use the different weights for positive and negative instances
        for i in range(0, n):
            if y[i] == 1:
                h[n + i] = self.c[0]
            else:
                h[n + i] = self.c[1]
        h = matrix(h)

        cvxopt.solvers.options['show_progress'] = False
        svm = cvxopt.solvers.qp(P, q, G, h)

        self.alpha = np.array(svm['x'])
        # create mask
        masked_alpha = ma.masked_outside(self.alpha, 0, 0)
        self.alpha = masked_alpha.data

        self.support_vectors = x[masked_alpha.mask[:,0], :]

        # labels of the support-vectors
        self.support_vector_labels = y[masked_alpha.mask[:,0]]

        # number of support vectors
        self.n_alpha = np.size(self.alpha, 0)

    def predict_plane(self, x):

        # size of x
        n_x = np.size(x, 0)

        # initialize result array
        f_x = np.zeros(n_x)
        for i in range(0, n_x):
            for j in range(0, self.n_alpha):
                f_x[i] = f_x[i] + self.alpha[j] * self.support_vector_labels[j] * self.kernel(
                    self.support_vectors[j, :], x[i, :])

            # element is in plus, when f[x] is positive  and in minus else
            if f_x[i] >= 0:
                f_x[i] = 1
            else:
                f_x[i] = -1
        return f_x

    def kernel(self, param, param1):
        # linear only for testing purposes
        if params.KERNEL_TYPE == 'linear':
            if np.transpose(param) @ param1 == 0:
                return 0.0001
            else:
                return np.transpose(param) @ param1
        elif params.KERNEL_TYPE == 'gaussian':
            # - 'gaussian': exp(-||x1-x2||^2/(2 sigma^2))
            return np.exp(-np.abs(np.transpose(param - param1) @ (param - param1)) / (2 * params.KERNEL_PARAM ** 2))
        elif params.KERNEL_TYPE == 'polynomial':
            # - 'polynomial': (x1^T x_2 +1)^p
            return (np.transpose(param) @ param1 + 1) ** params.KERNEL_PARAM
        else:
            IOError("Unknown kernel type: Allowed kernels are linear, gaussian and polynomial")

