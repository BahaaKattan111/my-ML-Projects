import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score

pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 2000)

# load the dataset
df = pd.read_csv('measurements.csv')
df.drop(['refill gas', 'refill liters', 'AC', 'rain', 'sun'], axis=1, inplace=True)

# preprocessing the data #
# dealing with missing values
df['specials'].fillna('Nothing', inplace=True)

# converting number from object to float and change ',' to '.'
cats = ['distance', 'consume', 'temp_inside']

for cat in df[['consume', 'temp_inside', 'distance']]:
	df[cat] = df[cat].str.replace(',', '.').astype(float)
print(df.info())
# dealing with categorical data
cat_list = ['specials', 'gas_type']
dummy_df = pd.get_dummies(df[cat_list])
df = pd.concat([df, dummy_df], axis=1)
df = df.drop(cat_list, axis=1)

# dealing with some missing values after converting the numbers to float
df['temp_inside'].fillna(df['temp_inside'].mean(), inplace=True)
na = (df.isnull().sum() / len(df)) * 100
# print(na)

# putting the values of X and y
X = df.drop('consume', axis=1)
y = df.consume

# separate X and y training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()

# normalizing the data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# fitting the model
model = ElasticNetCV(cv=3)
model.fit(X_train, y_train)

# evaluating the model
y_pred = model.predict(X_test)
print('mean squard error', mean_squared_error(y_test, y_pred))

val_score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
val_pred = cross_val_predict(model, X_train, y_train, cv=5)
print(f'training cost: {-1 * val_score.mean()}')
print(f'prediction cost: {val_pred.mean()}')
