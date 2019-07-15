import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from sklearn.model_selection import train_test_split
from numpy.linalg import eig
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score

data = load_iris()
y = data.target.reshape(-1,1)
x = data.data

def plot_PCA(xs, ys):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(xs)
    cov_mat = np.cov(X_std.T)      #covarience matrix (C = 1/n(X.X.T))
    values, vectors = eig(cov_mat)
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
#plot_PCA(x,y)

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
#plot_tSNE(x,y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

def forest_classifier_training(x_train, y_train):
    y_train = y_train.ravel()
    bagging_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random",
    max_leaf_nodes=16),n_estimators=500,max_samples=1.0,bootstrap=True,n_jobs=-1)
    bagging_clf.fit(x_train,y_train)
    return bagging_clf
clf = forest_classifier_training(x_train,y_train)

def cross_validation():
    pass


#y_hat = clf.predict(x_test)
#print("Score:", accuracy_score(y_hat,y_test))
