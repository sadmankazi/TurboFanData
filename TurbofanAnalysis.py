import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from math import sqrt

feat_names = ['Unit', 'CycleNo', 'opset1', 'opset2', 'opset3', 'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5',
              'sensor6', 'sensor7', 'sensor8', 'sensor9', 'sensor10', 'sensor11', 'sensor12', 'sensor13', 'sensor14',
              'sensor15', 'sensor16', 'sensor17','sensor18', 'sensor19','sensor20', 'sensor21','sensor22', 'sensor23',
              'sensor24', 'sensor25', 'sensor26']  #Add pythonic regilar expression

data_train1 = pd.read_csv('./CMAPSSData/train_FD001.txt', sep=' ', header=None, names=feat_names)
# data_test1 = pd.read_csv('./CMAPSSData/test_FD001.txt', sep = ' ', header = None)
# data_RUL1 = pd.read_csv('./CMAPSSData/RUL_FD001.txt', sep = ' ', header = None)

# data_train1.isna().sum() # Finding NaNs

# print(data_train1['Unit'].value_counts())

# plt.plot(data_train1['Unit'].value_counts()) # no of data points per unit
# plt.show()

data_train1['RUL'] = ""

max_cycle = data_train1.groupby('Unit')['CycleNo'].max().reset_index()
# print(max_cycle)
# print(max_cycle.loc[[1]])

# print(data_train1[data_train1['Unit'] == 1]['RUL'])

# Add checking for NaNs in all columns

# print(max_cycle.iloc[2, 1])

for x in max_cycle.iloc[:, 0]:
    count_cycle = max_cycle.iloc[x-1, 1]
    data_train1.loc[data_train1['Unit'] == x, 'RUL'] = count_cycle - data_train1['CycleNo']

df = data_train1.drop(['RUL', 'sensor22', 'sensor23', 'sensor25', 'sensor26', 'sensor24', 'Unit', 'CycleNo',
                       'opset1', 'opset2', 'opset3', 'sensor1', 'sensor5', 'sensor10', 'sensor18', 'sensor16', 'sensor19'], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled)

# print(df_normalized.head())

X_train, X_test, y_train, y_test = train_test_split(df_normalized, data_train1["RUL"], test_size=0.7, random_state=5)
leg = linear_model.LinearRegression()
leg.fit(X_train, y_train)
predictions = leg.predict(X_test)

# nn = MLPRegressor(hidden_layer_sizes=(15, 20), random_state=4)
# nn.fit(X_train, y_train)
# predictions = nn.predict(X_test)

mse = sqrt(mean_squared_error(y_test, predictions))
print('RMSE = {}'.format(round(mse, 3)))

# plot histogram of cycle lifetimes
plt.hist(max_cycle.iloc[:, 1], color='blue', edgecolor='black', bins=int(360/10), alpha=0.5)
plt.show()

plt.plot(data_train1['RUL'], data_train1['sensor7'], 'o', alpha=0.3)
plt.show()

plt.plot(data_train1['sensor3'], data_train1['sensor2'], 'o', alpha=0.3)
plt.show()

#Correlations
df = data_train1.drop(['RUL', 'sensor22', 'sensor23', 'sensor25', 'sensor26', 'sensor24', 'opset1', 'opset2', 'opset3','CycleNo','Unit'], axis=1)
correlations = df.corr()

# plot correlation matrix
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 21, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(list(range(1, 22)))
ax.set_yticklabels(list(range(1, 22)))
# plt.grid()
plt.show()

#plot all
plt.plot(data_train1['CycleNo'], df, '-')
plt.show()

# plt.plot(data_train1['CycleNo'],df_normalized,'-')
# plt.show()