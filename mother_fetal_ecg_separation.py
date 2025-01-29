import numpy as np
import matplotlib.pyplot as plt

# abdominal_ecg -> 5 signals, first one is scalp on the fetus ecg
# and the other 4 are abdominal ecg on the mother
# fetal_beat_positions -> positions of the fetal beats (used as a reference)
# sampling rate = 1000Hz
pos = np.loadtxt('fetal_beat_positions.csv', delimiter=',')
print(pos.shape)
Y = np.loadtxt('abdominal_ecg.csv', delimiter=',')
scalpECG = Y[:, 0]
# aY -> abdominal ecgs
aY = Y[:, 1:5]
aY = np.matrix.transpose(aY)

print(Y.shape)
print(aY.shape)

fix, axs = plt.subplots(5, 1, figsize=(16,20))
axs[0].plot(scalpECG)
axs[0].set_title('Scalp ECG')
for i in range(4):
  axs[i+1].plot(aY[i,:])
  axs[i+1].set_title(f"Abdominal ECG {i+1}")
  valid_pos = pos[pos < aY.shape[1]]
  axs[i+1].scatter(valid_pos, aY[i, valid_pos.astype(int)], color='red', label='fetal peaks positions')
plt.tight_layout()
plt.show()
print()

# Principal Component Analysis
covariance = np.cov(aY, rowvar=True)

val, V = np.linalg.eig(covariance)

# eigenvalues ordered in descending order
idx = np.argsort(val)[::-1]
val = val[idx]
V = V[:, idx]

# W -> unmixing matrix, which is the eigenvector matrix V transposed.
W = np.matrix.transpose(V)

# Sources extraction
Xpca = np.matmul(W, aY)

# Plot of the 4 estimated sources
figs, ax = plt.subplots(4, 1, figsize=(16, 20))
for j in range(4):
  ax[j].plot(Xpca[j,:])
  ax[j].set_title(f"PCA of source {j+1}")
plt.tight_layout()
plt.show()

# Variance of each estimated source
variances = np.var(Xpca, axis=1)
print(f"Variances: {variances}")
print(f"Eigenvalues: {val}")

normalized_val = val / np.sum(val)
tot_variance = np.cumsum(normalized_val)
num_components = np.argmax(tot_variance >= 0.90) + 1
print("Number of components which cover at least 90% of the overall variance: ", num_components)