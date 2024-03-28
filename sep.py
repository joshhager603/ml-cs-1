import pandas as pd

path = 'apple_quality.csv'
fulldata = pd.read_csv(path).to_numpy()
fulldata = fulldata[:-1, 1:]

for row in fulldata:
    if row[-1] == 'good':
        row[-1] = '1.0'
    else:
        row[-1] = '0.0'

path2 = './Data/train.csv'
traindata = pd.read_csv(path2).to_numpy()

fulldata = fulldata.tolist()
for i in range(len(fulldata)):
    for j in range(len(fulldata[i])):
        fulldata[i][j] = str(fulldata[i][j])

traindata = traindata.tolist()
for i in range(len(traindata)):
    for j in range(len(traindata[i])):
        traindata[i][j] = str(traindata[i][j])

testdata = []
for row in fulldata:
    if row not in traindata:
        testdata.append(row)
        
print(len(fulldata))
print(len(fulldata) - len(traindata))
print(len(testdata))

testdata = pd.DataFrame(testdata)
testdata.to_csv('test.csv')