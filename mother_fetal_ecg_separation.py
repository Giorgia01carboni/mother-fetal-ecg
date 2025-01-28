import numpy as np
import matplotlib.pyplot as plt

Y = np.loadtxt('abdominal_ecg.csv', delimiter=',')
scalpECG = Y[:, 0]
aY = Y[:, 1:5]
aY = np.matrix.transpose(aY)

print(Y.shape)
print(aY.shape)

fix, axs = plt.subplots(5, 1, figsize=(10,12))
axs[0].plot(scalpECG)
axs[0].set_title('Scalp ECG')
for i in range(4):
  axs[i+1].plot(aY[i,:])
  axs[i+1].set_title(f"Abdominal ECG {i+1}")
plt.tight_layout()
plt.show()


