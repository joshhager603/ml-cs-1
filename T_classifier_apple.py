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

#Standardize
x_scaler = preprocessing.StandardScaler().fit(X)
X = x_scaler.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Has not been preprocessed

def lr():
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    return clf.score(X_test, y_test)

def svm():
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

def dt():
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

print("Logistic Regression Score: " + str(lr()))
print("SVM Score: " + str(svm()))
print("Decision Tree Score: " + str(dt()))