import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, scorer
# from sklearn import preprocessing
# import seaborn as sns; sns.set()
# import time
import numpy as np
# from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.model_selection import RandomizedSearchCV
# import warnings

import sys
import os

os.chdir(sys.path[0]) # Change path to current directory

feat_names = ['Unit', 'CycleNo', 'opset1', 'opset2', 'opset3', 'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5',
              'sensor6', 'sensor7', 'sensor8', 'sensor9', 'sensor10', 'sensor11', 'sensor12', 'sensor13', 'sensor14',
              'sensor15', 'sensor16', 'sensor17','sensor18', 'sensor19','sensor20', 'sensor21','sensor22', 'sensor23',
              'sensor24', 'sensor25', 'sensor26']  #Add pythonic regilar expression

data_train1 = pd.read_csv('./CMAPSSData/train_FD001.txt', sep=' ', header=None, names=feat_names)
data_RUL1 = pd.read_csv('./CMAPSSData/RUL_FD001.txt', sep = ' ', header = None, usecols = [0], names=['RUL_actual'])

data_RUL1['Unit'] = data_RUL1.index + 1

# print(data_RUL1.head())
# print(data_train1.isna().sum()) # Finding NaNs
# print(data_train1.nunique()['Unit']) # Total number of units
print(data_train1.describe())

data_train1['RUL'] = ""

max_cycle = data_train1.groupby('Unit')['CycleNo'].max().reset_index()
# print(max_cycle.head())
# print(data_train1[data_train1['Unit'] == 1]['RUL'])

for x in max_cycle.iloc[:, 0]:
    count_cycle = max_cycle.iloc[x-1, 1]
    data_train1.loc[data_train1['Unit'] == x, 'RUL'] = count_cycle - data_train1['CycleNo']

# df_mean = data_train1.mean(axis=0) 
# print(df_mean)

# It doesn't appear these columns do much, so drop:
data_train1 = data_train1.drop(['sensor22', 'sensor23', 'sensor25', 'sensor26', 'sensor24', 'Unit',
                       'opset1', 'opset2', 'opset3', 'sensor1', 'sensor5', 'sensor10', 'sensor18', 'sensor16', 'sensor19'], axis=1)

# print(df.head())

# min_max_scaler = preprocessing.MinMaxScaler()
# np_scaled = min_max_scaler.fit_transform(df)
# df_normalized = pd.DataFrame(np_scaled)
# print(df_normalized.head())

X_train, X_test, y_train, y_test = train_test_split(data_train1.drop(['RUL'], axis=1), data_train1["RUL"], test_size=0.90, random_state=5)


LR = linear_model.LinearRegression()
LR.fit(X_train, y_train)
predictions = LR.predict(X_test)

# nn = MLPRegressor(hidden_layer_sizes=(15, 20), random_state=4)
# nn.fit(X_train, y_train)
# predictions = nn.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test, predictions))
print('RMSE = {}'.format(round(mse, 3)))

# plot histogram of cycle lifetimes
plt.hist(max_cycle.iloc[:, 1], color='blue', edgecolor='black', bins=int(360/10), alpha=0.5)
plt.show()

plt.plot(data_train1['RUL'], data_train1['sensor7'], 'o', alpha=0.3)
plt.show()

plt.plot(data_train1['sensor3'], data_train1['sensor2'], 'o', alpha=0.3)
plt.show()

# #Correlations
# df = data_train1.drop(['RUL', 'sensor22', 'sensor23', 'sensor25', 'sensor26', 'sensor24', 'opset1', 'opset2', 'opset3','CycleNo','Unit'], axis=1)
# correlations = df.corr()

# # plot correlation matrix
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, vmin=-1, vmax=1)
# fig.colorbar(cax)
# plt.title('Correlation between features')
# ticks = np.arange(0, 21, 1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(list(range(1, 22)))
# ax.set_yticklabels(list(range(1, 22)))
# # plt.grid()
# plt.show()

# now create new column with RUL being 1 or 0 before/after 20 cycles 

def add(RUL): 
    if RUL > 20:
        result =1
    else:
        result =0
    return result

data_train1['RUL_passfail'] = data_train1.apply(lambda row : add(row['RUL']), axis = 1)

print(data_train1.head())

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

X_train, X_test, y_train, y_test = train_test_split(df, data_train1['RUL_passfail'], test_size=0.75, random_state=5)

BC = AdaBoostClassifier()
# BC.fit(X_train, y_train)
# predictions = BC.predict(X_test)

cv_results = cross_val_score(BC, X_train, y_train, cv=5, scoring= 'accuracy')

print(cv_results.mean())


#plot all
# plt.plot(data_train1['CycleNo'], df, '-')
# plt.show()

# plt.plot(data_train1['CycleNo'],df_normalized,'-')
# plt.show()


# # Let's consider two models for now
# models = []
# models.append(('LR', LinearRegression()))
# models.append(('Bagging', RandomForestRegressor()))

# # models.append(('LASSO', Lasso()))
# # models.append(('EN', ElasticNet()))
# # models.append(('KNN', KNeighborsRegressor()))
# # models.append(('CART', DecisionTreeRegressor()))
# # models.append(('SVR', SVR()))

# results = []
# names = []
# times = []
# num_folds = 7
# seed = 5

# for name, model in models:
#     kfold = KFold(n_splits=num_folds, random_state=seed)
#     start = time.time()
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring= 'neg_mean_squared_error')
#     run_time = time.time()-start
#     results.append(cv_results)
#     names.append(name)
#     times.append(run_time)
#     msg = "%s: %f (%f) %f" % (name, cv_results.mean(), cv_results.std(), run_time)
#     print(msg)

# # Create Visual for people
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# # Add a plot of MSE vs Time labeled with different regression algorithms when we come back
# fig1 = plt.figure()
# fig1.suptitle('Time Comparison')
# ax1 = fig.add_subplot(111)
# plt.plot(times, 'x', ms = 8)
# ax1.set_xticklabels(names)
# plt.show()