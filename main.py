 # =======================================================================
# This file is part of the CS519_Project_4 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import regressors
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Housing dataset
print("***************")
print("Housing dataset")
print("***************")

# Preprocessing
# Reading the file
df = pd.read_csv("housing.data.txt", delim_whitespace=True)

# Defining the columns
df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

# Separating x and y
x = df.iloc[:, :-1].values
y = df["MEDV"].values

# Standardizing
sc_x = StandardScaler()
x_std = sc_x.fit_transform(x)

sc_y = StandardScaler()
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

# Splitting train and test data
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.3, random_state=0)
x_std_tr, x_std_ts, y_std_tr, y_std_ts = train_test_split(x_std, y_std, test_size=0.3, random_state=0)

# Running the algorithms
regressor_names = ["linear_reg", "ransac", "ridge", "lasso", "normal_equation", "decision_tree_reg"]
regressor = regressors.Regressors(max_trials=100, min_samples=50, residual_threshold=5.0, alpha=1.0, max_depth=3,
                                  seed=1, x_tr=x_tr, y_tr=y_tr, x_ts=x_ts)
regressor_std = regressors.Regressors(max_trials=100, min_samples=50, residual_threshold=5.0, alpha=1.0, max_depth=3,
                                  seed=1, x_tr=x_std_tr, y_tr=y_std_tr, x_ts=x_std_ts)

for regressor_name in regressor_names:
    print("\n=======================================")
    print("# Without Standardizing:")
    y_tr_pred, y_ts_pred = regressor.call("run_" + regressor_name)
    print("---------------------------------------")
    print("# With Standardizing:")
    y_std_tr_pred, y_std_ts_pred = regressor_std.call("run_" + regressor_name)

    error_train = mean_squared_error(y_tr, y_tr_pred)
    error_test = mean_squared_error(y_ts, y_ts_pred)
    error_std_train = mean_squared_error(y_std_tr, y_std_tr_pred)
    error_std_test = mean_squared_error(y_std_ts, y_std_ts_pred)

    print("---------------------------------------")
    print("# Mean Squared Error:")
    print(regressor_name + " MSE train: %.3f, test: %.3f" % (error_train, error_test))
    print(regressor_name + " STD MSE train: %.3f, test: %.3f" % (error_std_train, error_std_test))


# California Renewable Production dataset
print("\n\n\n***************************************")
print("California Renewable Production dataset")
print("***************************************")

# Preprocessing
# Reading the file
df = pd.read_csv("all_breakdown.csv", header=0)

# Removing the first column (timestamp)
df.drop(columns="TIMESTAMP", inplace=True)

# Removing the columns containing nan values
df.dropna(axis=1, inplace=True)

# Separating x and y
x = df.iloc[:, :-1].values
y = df["WIND TOTAL"].values

# Standardizing
sc_x = StandardScaler()
x_std = sc_x.fit_transform(x)

sc_y = StandardScaler()
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

# Splitting train and test data
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.3, random_state=0)
x_std_tr, x_std_ts, y_std_tr, y_std_ts = train_test_split(x_std, y_std, test_size=0.3, random_state=0)

# Running the algorithms
regressor_names = ["linear_reg", "ransac", "ridge", "lasso", "decision_tree_reg"]
regressor = regressors.Regressors(max_trials=100, min_samples=50, residual_threshold=5.0, alpha=1.0, max_depth=3,
                                  seed=1, x_tr=x_tr, y_tr=y_tr, x_ts=x_ts)
regressor_std = regressors.Regressors(max_trials=100, min_samples=50, residual_threshold=5.0, alpha=1.0, max_depth=3,
                                  seed=1, x_tr=x_std_tr, y_tr=y_std_tr, x_ts=x_std_ts)

for regressor_name in regressor_names:
    print("\n=======================================")
    print("# Without Standardizing:")
    y_tr_pred, y_ts_pred = regressor.call("run_" + regressor_name)
    print("---------------------------------------")
    print("# With Standardizing:")
    y_std_tr_pred, y_std_ts_pred = regressor_std.call("run_" + regressor_name)

    error_train = mean_squared_error(y_tr, y_tr_pred)
    error_test = mean_squared_error(y_ts, y_ts_pred)
    error_std_train = mean_squared_error(y_std_tr, y_std_tr_pred)
    error_std_test = mean_squared_error(y_std_ts, y_std_ts_pred)

    print("---------------------------------------")
    print("# Mean Squared Error:")
    print(regressor_name + " MSE train: %.3f, test: %.3f" % (error_train, error_test))
    print(regressor_name + " STD MSE train: %.3f, test: %.3f" % (error_std_train, error_std_test))

# Performance improvement
print("\n\n\n======================")
print("PERFORMANCE IMPROVEMENT")
clf = LassoCV(cv=5)
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(x, y)
n_features = sfm.transform(x).shape[1]
while n_features > 4:
    sfm.threshold += 0.1
    x_new = sfm.transform(x)
    n_features = x_new.shape[1]

# Standardizing
sc_x = StandardScaler()
x_std_new = sc_x.fit_transform(x_new)

sc_y = StandardScaler()
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

# Splitting train and test data
x_std_new_tr, x_std_new_ts, y_std_tr, y_std_ts = train_test_split(x_std_new, y_std, test_size=0.3, random_state=0)

# Regressor
regressor_std = regressors.Regressors(max_trials=100, min_samples=50, residual_threshold=5.0, alpha=1.0, max_depth=3,
                                  seed=1, x_tr=x_std_new_tr, y_tr=y_std_tr, x_ts=x_std_new_ts)

# Run regressor
for regressor_name in regressor_names:
    print("\n=======================================")
    print("# With Standardizing:")
    y_std_tr_pred, y_std_ts_pred = regressor_std.call("run_" + regressor_name)

    error_std_train = mean_squared_error(y_std_tr, y_std_tr_pred)
    error_std_test = mean_squared_error(y_std_ts, y_std_ts_pred)

    print("---------------------------------------")
    print("# Mean Squared Error:")
    print(regressor_name + " STD MSE train: %.3f, test: %.3f" % (error_std_train, error_std_test))

























