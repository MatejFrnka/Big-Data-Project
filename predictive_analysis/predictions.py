import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

df_red_uniform = pd.read_csv('red_data_uniform.csv')
df_white_uniform = pd.read_csv('white_data_uniform.csv')

df_red = df_red.set_index('id')
df_red_uniform = df_red_uniform.set_index('id')
df_white = df_white.set_index('id')
df_white_uniform = df_white_uniform.set_index('id')


df_red_low_uniform = df_red_uniform[df_red_uniform['price_category'] == 'low']
df_red_medium_uniform = df_red_uniform[df_red_uniform['price_category'] == 'medium']
df_red_high_uniform = df_red_uniform[df_red_uniform['price_category'] == 'high']

df_white_low_uniform = df_white_uniform[df_white_uniform['price_category'] == 'low']
df_white_medium_uniform = df_white_uniform[df_white_uniform['price_category'] == 'medium']
df_white_high_uniform = df_white_uniform[df_white_uniform['price_category'] == 'high']


df_red_uniform = df_red_uniform.drop('price_category', axis = 1)
df_red_low_uniform = df_red_low_uniform.drop('price_category', axis = 1)
df_red_medium_uniform = df_red_medium_uniform.drop('price_category', axis = 1)
df_red_high_uniform = df_red_high_uniform.drop('price_category', axis = 1)

df_white_uniform = df_white_uniform.drop('price_category', axis = 1)
df_white_low_uniform = df_white_low_uniform.drop('price_category', axis = 1)
df_white_medium_uniform = df_white_medium_uniform.drop('price_category', axis = 1)
df_white_high_uniform = df_white_high_uniform.drop('price_category', axis = 1)


red_corr_spearman_uniform = df_red_uniform.corr(method ='spearman')
red_cols_spearman_uniform = red_corr_spearman_uniform.index
red_corr_map_spearman_uniform = np.corrcoef(df_red_uniform[red_cols_spearman_uniform].values.T)

white_corr_spearman_uniform = df_white_uniform.corr(method ='spearman')
white_cols_spearman_uniform = white_corr_spearman_uniform.index
white_corr_map_spearman_uniform = np.corrcoef(df_white_uniform[white_cols_spearman_uniform].values.T)


X_red = df_red_uniform
X_red = X_red.drop('Rating_avg', axis = 1)
Y_red = X_red['Price'].values
X_red = X_red.drop('Price', axis = 1).values
X_red_train, X_red_test, Y_red_train, Y_red_test = train_test_split(X_red, Y_red, test_size = 0.20, random_state = 1337)

X_red_low = df_red_low_uniform
X_red_low = X_red_low.drop('Rating_avg', axis = 1)
Y_red_low = X_red_low['Price'].values
X_red_low = X_red_low.drop('Price', axis = 1).values
X_red_low_train, X_red_low_test, Y_red_low_train, Y_red_low_test = train_test_split(X_red_low, Y_red_low, test_size = 0.20, random_state = 1337)

X_red_medium = df_red_medium_uniform
X_red_medium = X_red_medium.drop('Rating_avg', axis = 1)
Y_red_medium = X_red_medium['Price'].values
X_red_medium = X_red_medium.drop('Price', axis = 1).values
X_red_medium_train, X_red_medium_test, Y_red_medium_train, Y_red_medium_test = train_test_split(X_red_medium, Y_red_medium, test_size = 0.20, random_state = 1337)

X_red_high = df_red_high_uniform
X_red_high = X_red_high.drop('Rating_avg', axis = 1)
Y_red_high = X_red_high['Price'].values
X_red_high = X_red_high.drop('Price', axis = 1).values
X_red_high_train, X_red_high_test, Y_red_high_train, Y_red_high_test = train_test_split(X_red_high, Y_red_high, test_size = 0.20, random_state = 1337)

X_white = df_white_uniform
X_white = X_white.drop('Rating_avg', axis = 1)
Y_white = X_white['Price'].values
X_white = X_white.drop('Price', axis = 1).values
X_white_train, X_white_test, Y_white_train, Y_white_test = train_test_split(X_white, Y_white, test_size = 0.20, random_state = 1337)

X_white_low = df_white_low_uniform
X_white_low = X_white_low.drop('Rating_avg', axis = 1)
Y_white_low = X_white_low['Price'].values
X_white_low = X_white_low.drop('Price', axis = 1).values
X_white_low_train, X_white_low_test, Y_white_low_train, Y_white_low_test = train_test_split(X_white_low, Y_white_low, test_size = 0.20, random_state = 1337)

X_white_medium = df_white_medium_uniform
X_white_medium = X_white_medium.drop('Rating_avg', axis = 1)
Y_white_medium = X_white_medium['Price'].values
X_white_medium = X_white_medium.drop('Price', axis = 1).values
X_white_medium_train, X_white_medium_test, Y_white_medium_train, Y_white_medium_test = train_test_split(X_white_medium, Y_white_medium, test_size = 0.20, random_state = 1337)

X_white_high = df_white_high_uniform
X_white_high = X_white_high.drop('Rating_avg', axis = 1)
Y_white_high = X_white_high['Price'].values
X_white_high = X_white_high.drop('Price', axis = 1).values
X_white_high_train, X_white_high_test, Y_white_high_train, Y_white_high_test = train_test_split(X_white_high, Y_white_high, test_size = 0.20, random_state = 1337)


print("\n")
print("AVERAGE RATING NOT INCLUDED")

print("\n")
print("Red wines:")
print("ALL PRICE CATEGORIES")

scaler_red = StandardScaler().fit(X_red_train)
rescaled_X_red_train = scaler_red.transform(X_red_train)
model_red = GradientBoostingRegressor(random_state=21, n_estimators=500)
model_red.fit(rescaled_X_red_train, Y_red_train)

rescaled_X_red_test = scaler_red.transform(X_red_test)
predictions_red = model_red.predict(rescaled_X_red_test)
print("MAE = ", mean_absolute_error(Y_red_test, predictions_red))
print("MSE = ", mean_squared_error(Y_red_test, predictions_red))

print("\n")
print("LOW PRICE CATEGORY")

scaler_red_low = StandardScaler().fit(X_red_low_train)
rescaled_X_red_low_train = scaler_red_low.transform(X_red_low_train)
model_red_low = GradientBoostingRegressor(random_state=21, n_estimators=500)
model_red_low.fit(rescaled_X_red_low_train, Y_red_low_train)

rescaled_X_red_low_test = scaler_red_low.transform(X_red_low_test)
predictions_red_low = model_red_low.predict(rescaled_X_red_low_test)
print("MAE = ", mean_absolute_error(Y_red_low_test, predictions_red_low))
print("MSE = ", mean_squared_error(Y_red_low_test, predictions_red_low))

print("\n")
print("MEDIUM PRICE CATEGORY")

scaler_red_medium = StandardScaler().fit(X_red_medium_train)
rescaled_X_red_medium_train = scaler_red_medium.transform(X_red_medium_train)
model_red_medium = GradientBoostingRegressor(random_state=21, n_estimators=500)
model_red_medium.fit(rescaled_X_red_medium_train, Y_red_medium_train)

rescaled_X_red_medium_test = scaler_red_medium.transform(X_red_medium_test)
predictions_red_medium = model_red_medium.predict(rescaled_X_red_medium_test)
print("MAE = ", mean_absolute_error(Y_red_medium_test, predictions_red_medium))
print("MSE = ", mean_squared_error(Y_red_medium_test, predictions_red_medium))

print("\n")
print("HIGH PRICE CATEGORY")

scaler_red_high = StandardScaler().fit(X_red_high_train)
rescaled_X_red_high_train = scaler_red_high.transform(X_red_high_train)
model_red_high = GradientBoostingRegressor(random_state=21, n_estimators=500)
model_red_high.fit(rescaled_X_red_high_train, Y_red_high_train)

rescaled_X_red_high_test = scaler_red_high.transform(X_red_high_test)
predictions_red_high = model_red_high.predict(rescaled_X_red_high_test)
print("MAE = ", mean_absolute_error(Y_red_high_test, predictions_red_high))
print("MSE = ", mean_squared_error(Y_red_high_test, predictions_red_high))


print("\n")
print("White wines:")
print("ALL PRICE CATEGORIES")

scaler_white = StandardScaler().fit(X_white_train)
rescaled_X_white_train = scaler_white.transform(X_white_train)
model_white = GradientBoostingRegressor(random_state=21, n_estimators=500)
model_white.fit(rescaled_X_white_train, Y_white_train)

rescaled_X_white_test = scaler_white.transform(X_white_test)
predictions_white = model_white.predict(rescaled_X_white_test)
print("MAE = ", mean_absolute_error(Y_white_test, predictions_white))
print("MSE = ", mean_squared_error(Y_white_test, predictions_white))

print("\n")
print("LOW PRICE CATEGORY")

scaler_white_low = StandardScaler().fit(X_white_low_train)
rescaled_X_white_low_train = scaler_white_low.transform(X_white_low_train)
model_white_low = GradientBoostingRegressor(random_state=21, n_estimators=500)
model_white_low.fit(rescaled_X_white_low_train, Y_white_low_train)

rescaled_X_white_low_test = scaler_white_low.transform(X_white_low_test)
predictions_white_low = model_white_low.predict(rescaled_X_white_low_test)
print("MAE = ", mean_absolute_error(Y_white_low_test, predictions_white_low))
print("MSE = ", mean_squared_error(Y_white_low_test, predictions_white_low))

print("\n")
print("MEDIUM PRICE CATEGORY")

scaler_white_medium = StandardScaler().fit(X_white_medium_train)
rescaled_X_white_medium_train = scaler_white_medium.transform(X_white_medium_train)
model_white_medium = GradientBoostingRegressor(random_state=21, n_estimators=500)
model_white_medium.fit(rescaled_X_white_medium_train, Y_white_medium_train)

rescaled_X_white_medium_test = scaler_white_medium.transform(X_white_medium_test)
predictions_white_medium = model_white_medium.predict(rescaled_X_white_medium_test)
print("MAE = ", mean_absolute_error(Y_white_medium_test, predictions_white_medium))
print("MSE = ", mean_squared_error(Y_white_medium_test, predictions_white_medium))

print("\n")
print("HIGH PRICE CATEGORY")

scaler_white_high = StandardScaler().fit(X_white_high_train)
rescaled_X_white_high_train = scaler_white_high.transform(X_white_high_train)
model_white_high = GradientBoostingRegressor(random_state=21, n_estimators=500)
model_white_high.fit(rescaled_X_white_high_train, Y_white_high_train)

rescaled_X_white_high_test = scaler_white_high.transform(X_white_high_test)
predictions_white_high = model_white_high.predict(rescaled_X_white_high_test)
print("MAE = ", mean_absolute_error(Y_white_high_test, predictions_white_high))
print("MSE = ", mean_squared_error(Y_white_high_test, predictions_white_high))
