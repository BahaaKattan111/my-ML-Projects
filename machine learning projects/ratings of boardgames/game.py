# Importing the Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# setting the display for terminal screen
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 2000)

# importing the data
df = pd.read_csv('board_data.csv')

# removing the useless features
df.drop(['users_rated', 'id', 'type', 'name', 'minplaytime', 'maxplaytime'], axis=1, inplace=True)

# removing the missing values
df.dropna(inplace=True)

# splitting the data to X and y
X = df.drop(['average_rating'], axis=1)
y = df['average_rating']

# searching for correlations between the features
'''corralation = X.corr()
fig = plt.figure(figsize=(10, 9))
sns.heatmap(corralation, vmax=0.8, square=True)
plt.show()'''

# splitting X and y to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# running the data using Linear Regresion model
reg = LinearRegression(normalize=True)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
acc1 = mean_squared_error(y_test, y_pred)

# running the data using Random Forest Regressor model
reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, min_samples_split=15, max_depth=20, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
acc2 = mean_squared_error(y_test, y_pred)

# running the data using LineaSVR model
reg = LinearSVR(C=10, epsilon=2)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
acc = mean_squared_error(y_test, y_pred)
acc3 = mean_squared_error(y_test, y_pred)

# evaluating the three models
print(f'''Linear regression accuracy:{acc3}|\nRandom Forest Regressor accuracy:{acc2}|\nLinearSCV accuracy:{acc1}|''')
