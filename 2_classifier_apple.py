import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

path = './Data/train.csv'
data = pd.read_csv(path).to_numpy()
X = data[:,0:-1]
y = data[:,-1]

# L2, L3, PCA, LDA

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

#Standardize
x_scaler = preprocessing.StandardScaler().fit(X)
X = x_scaler.transform(X)

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=0)


#Has not been preprocessed

def lr():
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    return clf.score(X_validate, y_validate)

def svm():
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    return clf.score(X_validate, y_validate)

def dt():
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    return clf.score(X_validate, y_validate)

print("Logistic Regression Train Accuracy: " + str(lr()))
print("SVM Train Accuracy: " + str(svm()))
print("Decision Tree Train Accuracy: " + str(dt()))