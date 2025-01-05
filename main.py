import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Data = pd.read_csv('kc_house_data.csv')
Data.head(5)

Data = Data.drop('date',axis=1)
Data = Data.drop('id',axis=1)
Data = Data.drop('zipcode',axis=1)

X = Data.drop('price',axis =1).values
y = Data['price'].values

print(y)

#splitting Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# Multiple Liner Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set result
y_pred = regressor.predict(X_test)

# # visualizing residuals
# fig = plt.figure(figsize=(10,5))
# residuals = (y_test- y_pred)
# # sns.distplot(residuals)

#compare actual output values with predicted values
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(10)
print(df1)


# # evaluate the performance of the algorithm (MAE - MSE - RMSE)
# from sklearn import metrics
# print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# print('MSE:', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('VarScore:',metrics.explained_variance_score(y_test,y_pred))