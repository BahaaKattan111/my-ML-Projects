# Importing the Libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, model_selection

# Importing the Data
data = pd.read_csv('car.data')

# Preprcessing the Data
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

# Splitting the Data to X and y
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# Training the model
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_train, y_train)

# Evaluating the Model
acc = model.score(X_test, y_test)
print(acc)
predicted = model.predict(X_test)
names = ['acc', 'unacc', 'good', 'vgood']

# Showing the Data
for X in range(len(predicted)):
	print('predicted:', names[predicted[X]], 'data:', X_test[X], 'actual', names[y_test[X]])
	n = model.kneighbors([X_test[X]], 9, True)
	print('N:', n)
