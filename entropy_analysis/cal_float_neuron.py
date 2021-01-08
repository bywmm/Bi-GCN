import numpy as np
import matplotlib.pyplot as plt


def get_entropy(vec, interval_cnt):
    ma = vec.max()
    mi = vec.min()
    interval = np.linspace(ma, mi, interval_cnt, endpoint=False)
    interval = np.flipud(interval)
    pdf_cnt = []
    cdf_cnt = [0]
    for pos, i in enumerate(interval):
        cdf_cnt.append((vec <= i).sum())
        pdf_cnt.append(cdf_cnt[pos+1] - cdf_cnt[pos])
    pdf_cnt = np.array(pdf_cnt)
    pdf = pdf_cnt / len(vec)
    entropy = (-1 * pdf * np.log2(pdf+1e-8)).sum()
    return entropy


x = np.load('data/pubmed_x.npy')
n, d = x.shape
# print('Load data: x')
# print(x, 'with shape ({:d}, {:d})'.format(n, d))

y0 = x[:, 0]

entropy = 0
for i in range(d):
    entropy += get_entropy(x[:, i], 200)
print(entropy)

union_entropy = np.log2(n) * 16
print('maximum entropy:', union_entropy)
binary_entropy = np.log2(2) * 16
print('union binary entropy:', binary_entropy)
# p0 = 0.2
# p1 = 1-p0
# binary_entropy = -(p0*np.log2(p0)+p1*np.log2(p1))
# print("binary entropy with p0={:.2f}:".format(p0), binary_entropy)
# print(np.sqrt(2708))
