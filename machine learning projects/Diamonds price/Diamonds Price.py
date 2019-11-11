# Importing the Libraries
import pandas as pd
import sklearn
from sklearn.svm import SVR
# Importing the Data
df = pd.read_csv('diamonds.csv', index_col=0)

# Preprocessing the Data
df['cut'] = df['cut'].map({"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5})
df['clarity'] = df['clarity'].map({"I3": 1, "I2": 2, "I1": 3, "SI2": 4, "SI1": 5, "VS2": 6, "VS1": 7, "VVS2": 8,
                                   "VVS1": 9, "IF": 10, "FL": 11})
df['color'] = df['color'].map({"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7})
df = sklearn.utils.shuffle(df)

# Splitting the Data to X and y
X = df.drop('price', axis=1).values
X = sklearn.preprocessing.scale(X)
y = df['price'].values

# Training the Model
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
clf = SVR(kernel='linear')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
for X, y in zip(X_test, y_test):
	print(f'Model:{clf.predict(X_test)[0]}, Actual:{y}')
