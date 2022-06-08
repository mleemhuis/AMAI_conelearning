# Parameters for the cone-based SVM
# Kernel-type:
# - 'linear': only for testing, it's more efficient to use KERNEL=False in this case
# - 'gaussian': exp(-||x1-x2||^2/(2 sigma^2))
# - 'polynomial': (x1^T x_2 + 1)^p
KERNEL_TYPE = 'gaussian'
# parameters (for gaussian and polynomial)
KERNEL_PARAM = 5

# incorrectness-value for the SVM (kernel-case)
C = 10

# Threshold for stopping criteria
TR = 10

# Parameters for classical SVM
KERNEL_PARAM_SVM = 0.0001

# incorrectness-value for the SVM (kernel-case)
C_SVM = 100
