import numpy as np
import matplotlib.pyplot as plt

pos = np.loadtxt('fetal_beat_positions.csv', delimiter=',')
print(pos.shape)
Y = np.loadtxt('abdominal_ecg.csv', delimiter=',')
scalpECG = Y[:, 0]
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
covariance = np.cov(aY)

val, V = np.linalg.eig(covariance)

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