import numpy as np

x = np.load('data/pubmed_x.npy')
n, d = x.shape
print('Load data, with the shape of ({:d}, {:d})'.format(n, d))

# per feature (input neuron)
x_sum0 = (x==0).sum(0)
x_sum1 = n - x_sum0
p1 = x_sum1 / n
p0 = x_sum0 / n
print("p1:", p1)
print('p0:', p0)


eps = 1e-8
input_entropy_0 = -1 * p0 * np.log2(p0+eps)
input_entropy_1 = -1 * p1 * np.log2(p1+eps)
input_entropy = input_entropy_0 + input_entropy_1
print("Information entropy contained in 1433 neurons: ")
print(input_entropy)
print("Total information entropy of feed data:")
print(input_entropy.sum())