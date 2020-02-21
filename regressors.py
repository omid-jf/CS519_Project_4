# ======================================================================= 
# This file is part of the CS519_Project_4 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

from time import time
import inspect
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import numpy as np


# This class will encapsulate all regressors
class Regressors(object):
    # Constructor
    def __init__(self, max_trials, min_samples, residual_threshold, alpha, max_depth, seed=1, x_tr=[], y_tr=[], x_ts=[]):
        self.max_trials = max_trials
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.alpha = alpha
        self.max_depth = max_depth
        self.seed = seed
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_ts = x_ts
        self.__obj = None

    def call(self, method):
        return getattr(self, method)()

    def __fit(self):
        start = int(round(time() * 1000))
        self.__obj.fit(self.x_tr, self.y_tr)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + " training time: " + str(end) + " ms")

    def __predict(self):
        start = int(round(time() * 1000))
        y_tr_pred = self.__obj.predict(self.x_tr)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + " prediction time of training data: " + str(end) + " ms")

        start = int(round(time() * 1000))
        y_ts_pred = self.__obj.predict(self.x_ts)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + " prediction time of testing data: " + str(end) + " ms")

        return y_tr_pred, y_ts_pred

    def run_linear_reg(self):
        self.__obj = LinearRegression()
        self.__fit()
        return self.__predict()

    def run_ransac(self):
        self.__obj = RANSACRegressor(LinearRegression(), min_samples=self.min_samples,
                                     residual_threshold=self.residual_threshold, max_trials=self.max_trials,
                                     random_state=self.seed)
        self.__fit()
        return self.__predict()

    def run_ridge(self):
        self.__obj = Ridge(alpha=self.alpha, random_state=self.seed)
        self.__fit()
        return self.__predict()

    def run_lasso(self):
        self.__obj = Lasso(alpha=self.alpha, random_state=self.seed)
        self.__fit()
        return self.__predict()

    def run_normal_equation(self):
        start = int(round(time() * 1000))
        onevec = np.ones((self.x_tr.shape[0]))
        onevec = onevec[:, np.newaxis]
        x_b = np.hstack((onevec, self.x_tr))

        w = np.zeros(self.x_tr.shape[1])
        z = np.linalg.inv(np.dot(x_b.T, x_b))
        w = np.dot(z, np.dot(x_b.T, self.y_tr))
        end = int(round(time() * 1000)) - start
        print("normal equation training time: " + str(end) + " ms")

        start = int(round(time() * 1000))
        y_tr_pred = np.dot(self.x_tr, w[1:]) + w[0]
        end = int(round(time() * 1000)) - start
        print("normal equation prediction time of training data: " + str(end) + " ms")

        start = int(round(time() * 1000))
        y_ts_pred = np.dot(self.x_ts, w[1:]) + w[0]
        end = int(round(time() * 1000)) - start
        print("normal equation prediction time of testing data: " + str(end) + " ms")

        return y_tr_pred, y_ts_pred

    def run_decision_tree_reg(self):
        self.__obj = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.seed)
        self.__fit()
        return self.__predict()
