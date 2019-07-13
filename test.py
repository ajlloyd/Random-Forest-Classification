import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

x,y = make_moons(n_samples=1000, noise=0.15, random_state=42)
print(x.shape, y.shape)

def plot_data(xs, ys):
    data = np.c_[xs,ys]
    plt.plot(data[y==0, 0], data[y==0, 1], "r.")
    plt.plot(data[y==1, 0], data[y==1, 1], "g.")
    plt.show()
plot_data(x,y)
