import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


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

def lr():
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    return clf.score(X_test, y_test)

def svm_tune():

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

    # fit the model using k-fold cv and grid search for hyperparam tuning
    clf = GridSearchCV(svc, hyperparams, cv=10, n_jobs=-1)
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
    tree = DecisionTreeClassifier(random_state=0)
    pca = PCA()
    pipe = Pipeline(steps=[("pca", pca), ("tree", tree)])
    param_grid = {
        "pca__n_components": range(1,8),
        "tree__min_samples_split": range(2,50)
    }
    search = GridSearchCV(pipe, param_grid, cv=10, n_jobs=-1)
    search.fit(X,y)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)



def main():
    svm_best_train_score = svm_best_train()
    svm_test_score = svm_test()

    print(f'''
BEST MODEL:
(insert hyperparams)
Training Accuracy:

Test Accuracy:

------------------------------------
Other models:

[Logistic Regression]
          
Hyperparams:
Training Accuracy:

Test Accuracy:
          

[SVM]
          
Hyperparams: C=6, gamma=0.5, kernel=rbf
Training Accuracy: {svm_best_train_score}

Test Accuracy: {svm_test_score}


[Decision Tree]

Hyperparams:
Training Accuracy:

Test Accuracy:
''')

print("Logistic Regression Accuracy: " + str(lr()))
#print("Decision Tree Accuracy: " + str(dt()))
main()
