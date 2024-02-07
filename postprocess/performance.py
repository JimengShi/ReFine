#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


def metrics_s1_t1(ws_threshold, time_index, inv_y, inv_yhat, test_errors):
    print("Peformance when water level is over {} ft \n".format(ws_threshold))

    print("------ MAE & RMSE ------")
    inv_y_over = inv_y[inv_y[:, 0] > ws_threshold]
    inv_yhat_over = inv_yhat[inv_y[:, 0] > ws_threshold]
    print('MAE = {}'.format(float("{:.6f}".format(mae(inv_y_over, inv_yhat_over)))))
    print('RMSE = {}'.format(float("{:.6f}".format(sqrt(mse(inv_y_over, inv_yhat_over))))), "\n")


    print("------ Max Errors (t+1 at S1) ------")
    over_ws_errors = test_errors[inv_y[:, 0] > ws_threshold]
    print("Max Error of Over Estimation:", over_ws_errors[:, time_index].max())
    print("Max Error of Under Estimation:", over_ws_errors[:, time_index].min())
    print("Max Abs Error of Under Estimation:", np.max(np.abs(over_ws_errors[:, time_index])), "\n")


    print("------ Time # (t+1 at S1) ------")
    print("Time# of Over Estimation:", np.count_nonzero(over_ws_errors[:, time_index] > 0))
    print("Time# of Under Estimation:", np.count_nonzero(over_ws_errors[:, time_index] < 0), "\n")
    

    print("------ Area (t+1 at S1) ------")
    over_area, under_area = 0, 0

    for err_idx in range(len(over_ws_errors)):
        if over_ws_errors[err_idx, time_index] > 0:
            over_area += over_ws_errors[err_idx, time_index]
        else:
            under_area += over_ws_errors[err_idx, time_index]

    print("Area of Over Estimation: {}".format(over_area))
    print("Area of Under Estimation: {}".format(under_area))
    
