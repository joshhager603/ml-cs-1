import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

path = './Data/train.csv'
data = pd.read_csv(path)

'''
- check for missing data
- standardize the data
- try L1 and L2 reg
- forward and backward feature selection
- PCA and LDA (change value of k)
- k fold cross validation
- hyperparameter tuning

Want >75% accuracy on test set
Maybe shoot for >85% or >90% on validation set?
'''

# check for missing data 
for row in data.isnull().to_numpy():
    for element in row:
        assert element == False

data = data.to_numpy()
X = data[:,0:-1]
y = data[:,-1]

# standardize
x_scaler = preprocessing.StandardScaler().fit(X)
X = x_scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

test_path = './Data/test.csv'
test_data = pd.read_csv(test_path).to_numpy()
test_X = test_data[:,0:-1]
test_y = test_data[:,-1]
test_x_scaler = preprocessing.StandardScaler().fit(test_X)
test_X = test_x_scaler.transform(test_X)

'''
Done: Hyperparameter tuning, L1, L2, none regularizers.
To do: Forward and backward feature selection, PCA and LDA (needed?), k-fold cross validation(I think it does this alr?)
'''
def lr():

    # REMOVE
    tpath = './DELETE/test_mod.csv'
    tdata = pd.read_csv(tpath).to_numpy()
    tX = tdata[:,0:-1]
    ty = tdata[:,-1]
    tx_scaler = preprocessing.StandardScaler().fit(tX)
    tX = tx_scaler.transform(tX)
    # -------------------------------
    pca = PCA()
    lr = LogisticRegression(random_state=0)
    pipe = Pipeline(steps=[("pca", pca), ("lr", lr)])
    lr__C_params = np.arange(0,10,0.1)
    lr__penalties = [None, 'l1', 'l2']
    lr__solver = ['saga']
    params = {'pca__n_components': np.arange(4,8), 
              'lr__C':lr__C_params, 
              'lr__penalty': lr__penalties, 
              'lr__solver': lr__solver}  
    clf = GridSearchCV(pipe, params, cv=10, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(clf.best_estimator_)

    # REMOVE
    print('Test accuracy: ' + str(clf.score(tX, ty)))
    # ------------
    return clf.score(X_train, y_train)

def lr_best_train():
    lr = LogisticRegression(C =0.2,random_state=0,solver='saga')
    lr.fit(X_train, y_test)
    return lr.score(X_test, y_test)

def lr_test():
    lr = LogisticRegression(C=0.2,random_state=0,solver='saga')
    lr.fit(X_train, y_train)
    return lr.score(test_X, test_y)

def svm():

    # REMOVE
    tpath = './DELETE/test_mod.csv'
    tdata = pd.read_csv(tpath).to_numpy()
    tX = tdata[:,0:-1]
    ty = tdata[:,-1]
    tx_scaler = preprocessing.StandardScaler().fit(tX)
    tX = tx_scaler.transform(tX)
    # -------------------------------

    C_params = [i for i in range(1, 50)]
    gamma_params = ['auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    hyperparams = {'C':C_params, 'gamma':gamma_params}
    svc = SVC(kernel='rbf', random_state=1)
    clf = GridSearchCV(svc, hyperparams, cv=10, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(clf.best_estimator_)

    # REMOVE
    print('Test accuracy: ' + str(clf.score(tX, ty)))
    # ------------

    return clf.score(X_test, y_test)

def dt():
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

print("Logistic Regression Accuracy: " + str(lr()))
#print("SVM  Accuracy: " + str(svm()))
#print("Decision Tree Accuracy: " + str(dt()))