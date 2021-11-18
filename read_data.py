import sys

# !conda install --yes --prefix {sys.prefix} nb_black
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import random
from tqdm.notebook import tqdm


from scipy import stats
import datetime

# dir(atd2020)

############## load data ############
def load_in_data():
    data_dir = atd2021.data
    # print(data_dir)
    results_dir = (atd2021.root / "../results").resolve()
    # print(results_dir)
    filename = data_dir / "City4_all.parquet.brotli"
    print(f"Accessing data file from {filename}.")
    city = filename.stem.split(".")[0].replace("_all", "")
    data = atd2020.utilities.read_data(filename)
    data_obs = data[data.Observed == 1].copy()
    data_obs["Date"] = data_obs["Timestamp"].dt.date
    return data, data_obs

############## detrend the data ############
# data_detrended = atd2020.detrend.detrend(data)
# data_obs_detrended = atd2020.detrend.detrend(data_obs)