import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
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

def lr():
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    return clf.score(X_test, y_test)

def svm():

    # REMOVE
    tpath = 'test_mod.csv'
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
print("SVM  Accuracy: " + str(svm()))
print("Decision Tree Accuracy: " + str(dt()))