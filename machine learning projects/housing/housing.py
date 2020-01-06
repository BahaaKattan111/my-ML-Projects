import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import , train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

pd.set_option('display.max_columns', 2222)
pd.set_option('display.width', 2222)
df = pd.read_csv('housing.csv')

'''
df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=df['population'] / 100, label='population'
        , c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
# plt.show()
'''
# to turn categorical data to numeric:
cat_list = ['ocean_proximity']
dummy_df = pd.get_dummies(df[cat_list])
df = pd.concat([df, dummy_df], axis=1)
df = df.drop(cat_list, axis=1)

# dealing with missing values
df['total_bedrooms'].fillna(df['total_bedrooms'].mean(), inplace=True)

# separating df to X and y
X = df.drop('median_house_value', axis=1)
y = df.median_house_value

# some feature engineering
df['bedroom_per_room'] = df['total_rooms'] / df['total_bedrooms']
df['households_per_population'] = df['population'] / df['households']
df.drop(['households', 'total_bedrooms'], axis=1, inplace=True)
print(df.head())
# looking for correlation
# sns.heatmap(df.corr(), annot=True, square=True)
# plt.show()

# separating X and y to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# fitting the model
model = KNeighborsRegressor()
model.fit(X_train, y_train)

# evaluating the model
y_pred = model.predict(X_test)
print('mean squard error', mean_squared_error(X_test, y_test))