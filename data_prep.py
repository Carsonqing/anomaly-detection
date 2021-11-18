############## import modules ############
import sys
# !conda install --yes --prefix {sys.prefix} nb_black
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import seaborn as sns
import brotli
import atd2020
import atd2021
import snappy
from scipy.spatial import distance_matrix
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


from atd2020 import __init__
from atd2020 import detector
from atd2020 import detrend
from atd2020 import metrics
from atd2020 import utilities
from random import sample
from sklearn import datasets, linear_model, metrics, utils, exceptions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score
import datetime
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    ConstantKernel,
)
import random
from scipy import stats
import datetime

def processData(x,y):
    # global x
    print(x+y+1)
    return(x+y+1)

def prime_factor(value):
    factors = []
    for divisor in range(2, value-1):
        quotient, remainder = divmod(value, divisor)
        if not remainder:
            factors.extend(prime_factor(divisor))
            factors.extend(prime_factor(quotient))
            break
        else:
            factors = [value]
    return factors

results=[]
def collect_results(result):
    results.append(result)

def standardize_self2(df, ycol="TotalFlow"):
    y = df[ycol]
    m = np.mean(y)
    std = np.std(y)
    df["TotalFlow_norm"] = (y - m) / std
    return df

def find_weekdays(weekday, prior):
    Weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    if prior == True:
        if weekday != "Sunday":
            res = Weekdays[Weekdays.index(weekday) + 1]
        elif weekday == "Sunday":
            res = "Monday"
    if prior == False:
        if weekday != "Monday":
            res = Weekdays[Weekdays.index(weekday) - 1]
        elif weekday == "Monday":
            res = "Sunday"
    return res



def test(i):
    global data
    return data.shape[1]+i
def get_otherwkd_list(Weekday):
    Weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday","Saturday","Sunday"]
    Weekday_idx = Weekdays.index(Weekday)
    if Weekday_idx<5:
        otherwkd_list = [elem for elem in Weekdays[0:5] if elem != Weekday ]
    else:
        otherwkd_list = [elem for elem in Weekdays[5:7] if elem != Weekday ]
    return otherwkd_list
    # [elem for elem in fractions if elem != fraction]


def data_prep_comb(ID,data_obs_for_combine, comb_hours, comb_wkd, hour_bef_aft):
    # print("Processing ID:", ID)
    # global data_obs_for_combine
    # res_combined = pd.DataFrame(columns=["ID", "Date", "Hour", "TotalFlow"])
    df_list = []
    data_obs_for_combine_groups = data_obs_for_combine.groupby(["ID"])
    temp_copy = data_obs_for_combine_groups.get_group(ID).copy()
    temp_copy["Date"] = temp_copy["Timestamp"].dt.date
    n_rows = temp_copy.shape[0]
    Weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    for i in range(n_rows):
        curr_row = temp_copy.iloc[i]
        date = curr_row["Date"]
        hour = curr_row["Hour"]
        if comb_hours == True:
            ################ combine nearby hours
            if hour >= 5 and hour <= 10:
                new_hours = [5, 6, 7, 8, 9, 10]
                new_hours.remove(hour)
                date = [date] * 6
            elif hour >= 11 and hour <= 16:
                new_hours = [11, 12, 13, 14, 15, 16]
                new_hours.remove(hour)
                date = [date] * 6
            elif hour >= 17 and hour <= 22:
                new_hours=[17, 18, 19, 20, 21, 22]
                new_hours.remove(hour)
                date = [date] * 6
            elif hour == 23:
                new_hours=[23, 0, 1, 2, 3, 4]
                new_hours.remove(hour)
                post_date = date + datetime.timedelta(days=1)
                date = [date]
                date.extend([post_date] * 5)
            elif hour >= 0 and hour <= 4:
                new_hours = [23, 0, 1, 2, 3, 4]
                new_hours.remove(hour)
                prev_date = [date - datetime.timedelta(days=1)]
                prev_date.extend([date] * 5)
                date = prev_date
            for j in range(len(new_hours)):
                df_list.append([curr_row["ID"],date[j],new_hours[j],curr_row["TotalFlow"]])
        
        if comb_wkd == True:
            ################ combine weekdays
            date = curr_row["Date"]
            weekday = date.strftime("%A")
            other_wkd = get_otherwkd_list(weekday)
            for day in other_wkd:
                diff = Weekdays.index(day) - Weekdays.index(weekday)
                date_new = date + datetime.timedelta(days=diff)
                df_list.append([curr_row["ID"],date_new,curr_row["Hour"],curr_row["TotalFlow"]])

        if hour_bef_aft == True:
            ################ combine one hour before and one hour after at the same day
            date = curr_row["Date"]
            hour = curr_row["Hour"]
            hours = [hour-1, hour+1]
            for h in hours:
                if h==-1:
                    h=23
                if h ==24:
                    h=1
                df_list.append([curr_row["ID"],date,h,curr_row["TotalFlow"]])
        
        # append the original record at last
        df_list.append([curr_row["ID"],curr_row["Date"],hour,curr_row["TotalFlow"]])


    df = pd.DataFrame(df_list, columns = ["ID", "Date", "Hour", "TotalFlow"])
 
        # finalize the combine data
    return df
 

# ############## load data ############
# data_dir = atd2021.data
# # print(data_dir)
# results_dir = (atd2021.root / "../results").resolve()
# # print(results_dir)

# filename = data_dir / "City4_all.parquet.brotli"
# print(f"Accessing data file from {filename}.")
# city = filename.stem.split(".")[0].replace("_all", "")
# data = atd2020.utilities.read_data(filename)
# data_obs = data[data.Observed == 1]

# ############## detrend the data ############
# # data_detrended = atd2020.detrend.detrend(data)
# data_obs_detrended = atd2020.detrend.detrend(data_obs)

# ############## normalize the data: ID+Hour ############
# # data_obs_detrended_norm = data_obs_detrended.groupby(["ID", "Hour"]).apply(standardize_self2)
# # data_obs_detrended_norm.rename(columns={"TotalFlow": "TotalFlow_old", "TotalFlow_norm": "TotalFlow"}, inplace=True)
# # data_obs_detrended_norm_clean["Date"] = [i.strftime("%A") for i in data_obs_detrended_norm_clean.Timestamp]
# # data_obs_detrended_norm_clean = data_obs_detrended_norm[
# #     ["ID", "Timestamp", "Weekday", "Hour", "TotalFlow","Anomaly","Fraction_Observed"]
# # ]

# ############## normalize the data: ID+Weekday ############
# # data_obs_detrended_norm = data_obs_detrended.groupby(["ID", "Weekday"]).apply(standardize_self2)
# # data_obs_detrended_norm.rename(columns={"TotalFlow": "TotalFlow_old", "TotalFlow_norm": "TotalFlow"}, inplace=True)
# # data_obs_detrended_norm_clean["Date"] = [i.strftime("%A") for i in data_obs_detrended_norm_clean.Timestamp]
# # data_obs_detrended_norm_clean = data_obs_detrended_norm[
#     # ["ID", "Timestamp", "Weekday", "Hour", "TotalFlow","Anomaly","Fraction_Observed"]
# # ]


# ############## select the data for training ############

# # data_obs_for_combine = data_obs_detrended_norm_clean
# data_obs_for_combine = data_obs_detrended[["ID", "Timestamp","Weekday", "Hour", "TotalFlow","Anomaly","Fraction_Observed"]]
# df = data_prep_comb(0,data_obs_for_combine, False, True, True)
# print(df.iloc[0:10,])