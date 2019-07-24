import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import cross_val_predict,learning_curve
from numpy.linalg import eig
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn

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

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33,random_state=42)
y_train = y_train.ravel()

def forest_classifier_training(x_train, y_train):
    bagging_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random",
    max_leaf_nodes=16),n_estimators=500,max_samples=1.0,bootstrap=True,n_jobs=-1)
    bagging_clf.fit(x_train,y_train)
    return bagging_clf
clf = forest_classifier_training(x_train,y_train)

def grid_search(x_train, y_train, classifier):
    parameters = {"splitter":("best","random"), "max_leaf_nodes":[np.random.randint(2,high=50)], "random_state":[42]}
    gs_clf = RandomizedSearchCV(classifier.base_estimator_,parameters,cv=10)
    gs_clf.fit(x_train,y_train)
    best_clf=gs_clf.best_estimator_
    print(best_clf)
    return best_clf
best_clf = grid_search(x_train, y_train, clf)

def cross_validation(x_train, y_train, classifier):
    scores = cross_val_score(classifier, x_train, y_train,cv=10)
    print("All CV scores:", scores)
    print("Mean CV score:", scores.mean())
cross_validation(x_train,y_train,best_clf)

def learning_curves(estimator, x, y, train_sizes, cv):
    #Shows train/val scores of an estimator for varying no. of training samples.
    #Shows if there a benefit from adding more training data
    #Shows whether the estimator suffer from a variance error or a bias error
    train_sizes, train_scores, validation_scores = learning_curve(
    estimator, x, y, train_sizes = train_sizes,cv = cv,
    scoring = 'accuracy')
    train_scores_mean = np.mean(train_scores,axis=1)
    validation_scores_mean = np.mean(validation_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    validation_scores_std = np.std(validation_scores,axis=1)
    #Graph:
    plt.plot(train_sizes, train_scores_mean, "r-", label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, "g-", label = 'Validation (test) error')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
    validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    plt.ylabel('Accuracy Score', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize = 12)
    plt.ylim(0,1.1)
    plt.legend()
    plt.show()
    #not enough data for validation curves to give valid results
#learning_curves(best_clf, x_train, y_train, np.linspace(0.1,1.0,30), cv=30)

def plot_confusion_matrix(clf, x, y_actual, cv, title=None):
    y_pred = cross_val_predict(clf, x, y_actual, cv=cv)
    con_matrix = confusion_matrix(y_actual, y_pred)
    heatmap = sn.heatmap(con_matrix, annot=True)
    plt.title("Confusion Matrix of y_{}".format(title), fontsize = 12)
    plt.ylabel('Actual Label', fontsize = 14)
    plt.xlabel('Predicted Label', fontsize = 14)
    N = np.arange(0.5, 3)
    plt.xticks(N,('Setosa', 'Versicolour', "Virginica"))
    plt.yticks(N,('Setosa', 'Versicolour', "Virginica"))
    plt.show()
    print("Score:", accuracy_score(y_pred,y_actual))
plot_confusion_matrix(best_clf, x_train, y_train, 3, title="train")
#plot_confusion_matrix(best_clf, x_test, y_test, 3, title="test")
