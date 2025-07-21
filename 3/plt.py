import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("loss.csv", delimiter=",")
epochs = data[:,0]
loss = data[:,1]

plt.plot(epochs, loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.savefig("loss_curve.png")
