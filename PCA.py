import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from sklearn.model_selection import train_test_split


from sklearn.datasets import load_iris
data = load_iris()
y = data.target
x = data.data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



#PCA plot:
def plot_PCA(xs, ys, n_dimensions):
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)
    covarience_x = np.cov(scaled_x)       #covarience matrix (variance between adjacent pairs in matrix x)
    u,s,vT = svd(covarience_x)
    print(u, s.round(decimals=2), vT)


plot_PCA(x,y,n_dimensions=2)
