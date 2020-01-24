import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
# import the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
kaggle = pd.read_csv('sample_submission.csv')


print(train.info())
# the data is already cleaned!

# let`s separate the data in X & y
X = train.drop('label', axis=1)
y = train.label
# splitting the data into training sets and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=99)
'''
# fitting the model with Extra Tree Classifier
tree = ExtraTreesClassifier()

# looking for good parameters
params = {'n_estimators': [40, 80, 100, 200, 300], 'min_samples_leaf': [3, 6, 9], 'max_features': [1.0, 'auto', 0.7, 0.4],
          'min_samples_split': [3, 5, 7], 'max_depth': [7, 10, 15, 20, 25], 'max_leaf_nodes': [5, 10, 15],
          'criterion': ['entropy', 'gini']}

# training the data on small instances to avoid wasting time(days)
sub_Xtrain = X_train[0:7000]
sub_ytrain = y_train[0:7000]

# run GridSearch (looking for the good params)
grid = GridSearchCV(ExtraTreesClassifier(), param_grid=params, scoring='accuracy', cv=3)
grid.fit(sub_Xtrain, sub_ytrain)
print('Extra tree classifier`s best score', grid.best_score_, 'best params', grid.best_params_)

# fitting the optimized ExtraTreeClassifier by AdaBoostClassifier
model = AdaBoostClassifier(base_estimator=grid.best_estimator_, learning_rate=0.1, n_estimators=200, algorithm='SAMME')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
# the accuracy(f1_score) will be 0.96
'''
# now let`s try another model! XGBoost Classifier
model = XGBClassifier(max_depth=26, n_estimators=300, gamma=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
# the accuracy(f1_score) will be 0.97
y_pred = model.predict(test)

# now let`s submit to kaggle competition
predicted_test_value = pd.DataFrame({'ImageID': kaggle.ImageId,
                                     'Label': y_pred})

predicted_test_value.to_csv("PredictedTestScore_X.csv", index=False)
