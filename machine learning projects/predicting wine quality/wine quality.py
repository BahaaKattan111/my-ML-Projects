import matplotlib.pyplot as plt
from  sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

wine = pd.read_csv('winequality-red.csv')

# Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)

# seperate the dataset as response variable and feature variables
X = wine.drop('quality', axis=1)
y = wine['quality']
print(wine.head())

# Train and Test splitting of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying Standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# fitting the model
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
# print(classification_report(y_test, pred_rfc))

rfc_eval = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=23)
# print(rfc_eval.mean())

sns.barplot(data=wine, x='quality', y='citric acid')
plt.show()
