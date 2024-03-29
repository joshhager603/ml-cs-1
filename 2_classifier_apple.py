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

    # REMOVE
    print('Test accuracy: ' + str(clf.score(tX, ty)))
    # ------------
    return clf.score(X_train, y_train)

def lr_best_train():
    lr = LogisticRegression(C =0.2,random_state=0,solver='saga')
    lr.fit(X_train, y_train)
    return lr.score(X_test, y_test)

def lr_test():
    lr = LogisticRegression(C=0.2,random_state=0,solver='saga')
    lr.fit(X_train, y_train)
    return lr.score(test_X, test_y)

def svm_tune():

    # REMOVE
    tpath = './DELETE/test_mod.csv'
    tdata = pd.read_csv(tpath).to_numpy()
    tX = tdata[:,0:-1]
    ty = tdata[:,-1]
    tx_scaler = preprocessing.StandardScaler().fit(tX)
    tX = tx_scaler.transform(tX)
    # -------------------------------

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

    # REMOVE
    print('Test accuracy: ' + str(clf.score(tX, ty)))
    # ------------

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
    # REMOVE
    tpath = './DELETE/test_mod.csv'
    tdata = pd.read_csv(tpath).to_numpy()
    tX = tdata[:,0:-1]
    ty = tdata[:,-1]
    tx_scaler = preprocessing.StandardScaler().fit(tX)
    tX = tx_scaler.transform(tX)
    # -------------------------------

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
    
    # Test accuracy - remove
    print("Test accuracy " + str(search.score(tX, ty)))

    return clf.score(X_test, y_test)

def dt_best_train():
    dt = Pipeline(steps=[('pca', PCA(n_components=7)),
                ('tree', DecisionTreeClassifier(random_state=0))])
    dt.fit(X_train, y_train)
    return dt.score(X_test, y_test)

def dt_test():
    dt = Pipeline(steps=[('pca', PCA(n_components=7)),
                ('tree', DecisionTreeClassifier(random_state=0))])
    dt.fit(X_train, y_train)
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
Training Accuracy: {svm_best_train_score}

Test Accuracy: {svm_test_score}

------------------------------------
Other models:

[Logistic Regression]
          
Hyperparams: C=0.2, solver=saga
Training Accuracy: {lr_best_train_score}

Test Accuracy: {lr_test_score}

[Decision Tree]

Hyperparams: PCA__n_components=7
Training Accuracy: {dt_best_train_score}

Test Accuracy: {dt_test_score}
''')
    
main()
