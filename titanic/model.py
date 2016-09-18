import pandas as pd
from sklearn import cross_validation as cv
from sklearn import svm, linear_model, metrics, ensemble
from sklearn.naive_bayes import GaussianNB

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# list of models to test for best CV average
svm_model = svm.SVC(kernel='linear', C=1)
lr_model = linear_model.LogisticRegression()
rf_model = ensemble.RandomForestClassifier()
nb_model = GaussianNB()

target = train['Survived'].values
data = train

# serialize sex vector
data.loc[data.Sex == 'male','Sex'] = 1
data.loc[data.Sex == 'female','Sex'] = 0

# populate NaN values with average for vector
data['Fare'].fillna(data['Fare'].mean(),inplace=True)
data['Age'].fillna(data['Age'].mean(),inplace=True)

# set float vectors to int
data['Fare'] = data['Fare'].astype(int)
data['Age'] = data['Age'].astype(int)

# remove unused columns
data.drop(['Survived','Ticket','Name','Cabin','Embarked'],inplace=True,axis=1)

data_array = data.values

scores1 = cv.cross_val_score(svm_model,data_array,target,cv=10,scoring='accuracy')
scores2 = cv.cross_val_score(lr_model,data_array,target,cv=10,scoring='accuracy')
scores3 = cv.cross_val_score(rf_model,data_array,target,cv=10,scoring='accuracy')
scores4 = cv.cross_val_score(nb_model,data_array,target,cv=10,scoring='accuracy')

print(scores1.mean())
print(scores2.mean())
print(scores3.mean())
print(scores4.mean())

rf_model_fitted = rf_model.fit(data_array,target)
lr_model_fitted = lr_model.fit(data_array,target)

data = test
data.loc[data.Sex == 'male','Sex'] = 1
data.loc[data.Sex == 'female','Sex'] = 0
data['Fare'].fillna(data['Fare'].mean(),inplace=True)
data['Age'].fillna(data['Age'].mean(),inplace=True)
data['Fare'] = data['Fare'].astype(int)
data['Age'] = data['Age'].astype(int)
data.drop(['Ticket','Name','Cabin','Embarked'],inplace=True,axis=1)
test_array = data.values

# prediction = rf_model_fitted.predict(test_array)
prediction = lr_model_fitted.predict(test_array)

submission = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':prediction}).set_index('PassengerId').to_csv("submission.csv")
