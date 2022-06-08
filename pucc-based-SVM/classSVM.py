import numpy as np

def classSVM(tree, x):
    n = np.size(x, 0)
    classification = np.zeros(n)
    for i in range(0, n):
        classification[i] = tree.classify(x[i, :])

    return classification
