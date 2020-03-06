import numpy as np
from scipy.stats import wasserstein_distance

# documentation says this only works in the 1D case
# suppose we have an HMM with 4 states and it estimate a length-4 array 
# for the distribution over hidden states

prediction = np.array([1., 0., 0., 0.])
target = np.array([0., 0., 0., 1.])

print(wasserstein_distance(prediction, target))  # distance = 0
# debugging: this method is symmetric as expected
print(wasserstein_distance(target, prediction)) # distance = 0

print(wasserstein_distance(target, target))  # identity case, distance = 0