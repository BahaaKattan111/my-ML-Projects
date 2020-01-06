import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 2222)
pd.set_option('display.width', 2222)

df = pd.read_csv('student-mat.csv', sep=";")
print(df.head())

# looking for correlation
# sns.barplot(x='paid', y='G3', data=df)
# plt.show()

# preprocessing categorical data
for col in df[['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']]:
	map = {'yes': 1, 'no': 0}
	df[col] = df[col].map(map)
for col in df[['sex', 'Pstatus', 'address', 'Mjob', 'guardian', 'reason', 'Fjob', 'famsize', 'school']]:
	df[col] = pd.get_dummies(df[col], prefix=col)
	df.drop([col], axis=1, inplace=True)

print(df.head())
# separating data to X and y sets
X = df.drop('G3', axis=1)
y = df.G3

# separating X and y to training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=43)

# fitting the modol
model = SVC()
model.fit(Xtrain, ytrain)

# evaluating the model
ypred = model.predict(Xtest)
print(mean_squared_error(ytest, ypred))

