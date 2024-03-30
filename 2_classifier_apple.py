import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

'''
======================================================================================================
-- README --
CSDS 340
Case Study 1 Python Code
Group 2
Sam King (sjk171), Calvin Cai (cyc44), Josh Hager (jrh236)

Before running the code, please ensure the files ./Data/train.csv and ./Data/test.csv exist and 
are in the proper directory.

The code can be run using the command:
python3 2_classifier_apple.py
======================================================================================================
'''

path = './Data/train.csv'
data = pd.read_csv(path)

# check for missing data 
for row in data.isnull().to_numpy():
    for element in row:
        assert element == False

data = data.to_numpy()
X = data[:,0:-1]
y = data[:,-1]

# standardize the training dataset
x_scaler = preprocessing.StandardScaler().fit(X)
X = x_scaler.transform(X)

# split the training dataset into training and testing folds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# parse in test.csv and standardize the data
test_path = './Data/test.csv'
test_data = pd.read_csv(test_path).to_numpy()
test_X = test_data[:,0:-1]
test_y = test_data[:,-1]
test_x_scaler = preprocessing.StandardScaler().fit(test_X)
test_X = test_x_scaler.transform(test_X)

def lr():
    pca = PCA()
    lr = LogisticRegression(random_state=0)
    pipe = Pipeline(steps=[("pca", pca), ("lr", lr)])

    lr__C_params = np.arange(0.1,10,0.1)
    lr__penalties = [None, 'l1', 'l2']
    lr__solver = ['saga']
    params = {'pca__n_components': np.arange(4,8), 
              'lr__C':lr__C_params, 
              'lr__penalty': lr__penalties, 
              'lr__solver': lr__solver}  
    
    clf = GridSearchCV(pipe, params, cv=10, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(clf.best_estimator_)

    return clf.score(X_train, y_train)

def lr_best_train():
    lr = LogisticRegression(C =0.2,random_state=0,solver='saga')
    lr.fit(X_train, y_train)
    return lr.score(X_test, y_test)

def lr_test():
    lr = LogisticRegression(C=0.2,random_state=0,solver='saga')

    # train on the training data from train.csv
    lr.fit(X_train, y_train)

    # output the accuracy score from predicting the testing data from test.csv
    return lr.score(test_X, test_y)

def svm_tune():

    svc = SVC(kernel='rbf', random_state=1)
    pca = PCA()
    pipe = Pipeline(steps=[("pca", pca), ("svc", svc)])

    pca__n_components = range(4, 8)
    svc__C_params = [i for i in range(1, 50)]
    svc__gamma_params = ['auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # pca didn't do anything
    params = {
        'pca__n_components': pca__n_components,
        'svc__C': svc__C_params, 
        'svc__gamma': svc__gamma_params
     }

    # fit the model using k-fold cv and grid search for hyperparam tuning
    clf = GridSearchCV(pipe, params, cv=10, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(clf.best_estimator_)

    print("SVM Train Accuracy: " + str(clf.score(X_test, y_test)))

def svm_best_train():
    svc = SVC(kernel='rbf', C=6, gamma=0.5, random_state=1)
    svc.fit(X_train, y_train)
    return svc.score(X_test, y_test)

def svm_test():
    svc = SVC(kernel='rbf', C=6, gamma=0.5, random_state=1)

    # train the SVM on the training data from train.csv
    svc.fit(X_train, y_train)

    # output the accuracy score from predicting the testing data from test.csv
    return svc.score(test_X, test_y)

def dt():
    tree = DecisionTreeClassifier(random_state=0)
    pca = PCA()
    pipe = Pipeline(steps=[("pca", pca), ("tree", tree)])
    param_grid = {
        "pca__n_components": range(2,8),
    }
    search = GridSearchCV(pipe, param_grid, cv=10, n_jobs=-1)
    search.fit(X_train, y_train)
    clf = search.best_estimator_    
    print(clf)

    return clf.score(X_test, y_test)

def dt_best_train():
    dt = Pipeline(steps=[('pca', PCA(n_components=7)),
                ('tree', DecisionTreeClassifier(random_state=0))])
    dt.fit(X_train, y_train)
    return dt.score(X_test, y_test)

def dt_test():
    dt = Pipeline(steps=[('pca', PCA(n_components=7)),
                ('tree', DecisionTreeClassifier(random_state=0))])
    
    # train on the training data from train.csv
    dt.fit(X_train, y_train)

    # output the accuracy score from predicting the testing data from test.csv
    return dt.score(test_X, test_y)

def main():
    svm_best_train_score = svm_best_train()
    svm_test_score = svm_test()

    lr_best_train_score = lr_best_train()
    lr_test_score = lr_test()
    
    dt_best_train_score = dt_best_train()
    dt_test_score = dt_test()

    print(f'''
BEST MODEL:
[SVM]
          
Hyperparams: C=6, gamma=0.5, kernel=rbf
Training Accuracy: {round(svm_best_train_score * 100, 2)}%

Test Accuracy: {round(svm_test_score * 100, 2)} %

------------------------------------
Other models:

[Logistic Regression]
          
Hyperparams: C=0.2, solver=saga
Training Accuracy: {round(lr_best_train_score * 100, 2)}%

Test Accuracy: {round(lr_test_score * 100, 2)}%

[Decision Tree]

Hyperparams: PCA__n_components=7
Training Accuracy: {round(dt_best_train_score * 100, 2)}%

Test Accuracy: {round(dt_test_score * 100, 2)}%
''')
    
main()
