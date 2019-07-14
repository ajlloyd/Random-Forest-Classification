import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from sklearn.model_selection import train_test_split
from numpy.linalg import eig


from sklearn.datasets import load_iris
data = load_iris()
y = data.target
x = data.data



#PCA plot:
def plot_PCA(xs, ys):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(xs)
    cov_mat = np.cov(X_std.T)      #covarience matrix (C = 1/n(X.X.T))
    values, vectors = eig(cov_mat)
    print(vectors.shape)
    z = vectors.T.dot(X_std.T).T
    PCA1 = z[:,0]
    PCA2 = z[:,1]
    plt.plot(PCA1[ys==0],PCA2[ys==0],"g.")
    plt.plot(PCA1[ys==1],PCA2[ys==1],"b.")
    plt.plot(PCA1[ys==2],PCA2[ys==2],"r.")
    plt.show()

plot_PCA(x,y)
