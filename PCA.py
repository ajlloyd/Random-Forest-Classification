import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from sklearn.model_selection import train_test_split
from numpy.linalg import eig
from sklearn.manifold import TSNE


from sklearn.datasets import load_iris
data = load_iris()
y = data.target
x = data.data

print(data)



def plot_PCA(xs, ys):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(xs)
    cov_mat = np.cov(X_std.T)      #covarience matrix (C = 1/n(X.X.T))
    values, vectors = eig(cov_mat)
    print(vectors.shape)
    z = vectors.T.dot(X_std.T).T
    PCA1 = z[:,0]
    PCA2 = z[:,1]

    ax = plt.subplot(111)
    ax.plot(PCA1[ys==0],PCA2[ys==0],"g.",label="Iris-Setosa")
    ax.plot(PCA1[ys==1],PCA2[ys==1],"b.", label="Iris-Versicolour")
    ax.plot(PCA1[ys==2],PCA2[ys==2],"r.", label="Iris-Virginica")
    ax.legend()
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.show()
plot_PCA(x,y)


def plot_tSNE(xs,ys):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(xs)
    tSNE = TSNE(n_components=2)
    z = tSNE.fit_transform(X_std)
    tSNE1 = z[:,0]
    tSNE2 = z[:,1]

    ax = plt.subplot(111)
    ax.plot(tSNE1[ys==0],tSNE2[ys==0],"g.",label="Iris-Setosa")
    ax.plot(tSNE1[ys==1],tSNE2[ys==1],"b.", label="Iris-Versicolour")
    ax.plot(tSNE1[ys==2],tSNE2[ys==2],"r.", label="Iris-Virginica")
    ax.legend()
    plt.xlabel("tSNE1")
    plt.ylabel("tSNE2")
    plt.show()

plot_tSNE(x,y)
