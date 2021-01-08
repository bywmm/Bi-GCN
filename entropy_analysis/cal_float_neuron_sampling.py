# parzen window analyse for hidden units
import numpy as np

x = np.load('data/pubmed_x.npy')
n, d = x.shape
# print('Load data: x')


def Parzen(Data, X, h):
    Probs = []
    n = Data.shape[0]
    for x in X:
        dis = np.abs(Data-x)
        Probs.append((dis < h).sum() / n / h)
    return Probs

y0 = x[:,0]
X = np.linspace(y0.min(), y0.max(), 1000)
Probs = Parzen(y0, X, 0.1)
# plt.plot(X, Probs)
# plt.show()
# Probs = np.array(Probs)
# print(Probs)
# print(-1 * np.log2(Probs + 1e-8))
# plt.plot(X, -1 * Probs*np.log2(Probs+1e-8))
# plt.show()
# print(np.log2(2708))

##########Monte Carlo##########

# 1.概率密度probs跟谁比？y需要确定一下范围
# 2.窗口大小h
def Monte_Carlo(arr, num, max_y=6):
    samples = []
    while len(samples) < num:
        # randomly sample x in range[min_value, max_value] of neuron_arr
        x = np.random.random(num) * (arr.max() - arr.min()) + arr.min()
        y = np.random.random(num)  * max_y
        # print(arr.max() - arr.min())
        probs = Parzen(arr, x, 0.1)
        # print(probs)
        # assert False
        # print(probs)
        for i in range(num):
            # reject or not this sampled point.
            if y[i] < probs[i]:
                if len(samples) >= num:
                    break
                samples.append(x[i])
    return samples


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

M = 200
sample_sizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
ans = []
# sample_size = 20000 # 2708
for tmp in sample_sizes:
    sample_size = tmp * 1000
    entropy = 0
    for i in range(d):
        tmp_vec = Monte_Carlo(x[:, i], sample_size)
        entropy += get_entropy(np.array(tmp_vec), M)
    print("entropy: {:.4f}, sample size: {:d}, M: {:d}".format(entropy, sample_size, M))
    ans.append(entropy)
print(ans)

# print(16 * np.log2(200))

# 500000 72.62937020676578
# 100000 72.92154791782204
# 10000  76.6019469918534
# 1000   81.09890493516441
