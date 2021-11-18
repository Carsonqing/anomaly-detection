import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from imblearn.over_sampling import RandomOverSampler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import datetime
from sklearn import datasets, linear_model, metrics, utils, exceptions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import random
import math
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

start_date = datetime.date(2011, 1, 1)
end_date = datetime.date(2020, 12, 31)
delta = datetime.timedelta(days=1)
dates = []
# t = []
# counter = 1
while start_date <= end_date:
    dates.append(start_date)
    #     t.append(counter)
    start_date += delta
#     t += 1
sample_df = pd.DataFrame({"dates": dates})
sample_df["Weekday"] = pd.to_datetime(sample_df["dates"]).apply(
    lambda x: x.strftime("%A")
)
# sample_df

def get_dates(Weekday):
    global sample_df
    dates = []
    for day in Weekday:
        temp = sample_df.dates[sample_df.Weekday == day].tolist()
        #         print(type(temp))
        #         print(temp)
        dates += temp
        #         print(dates)
    #         print(dates)
    dates = sorted(dates)
    length = len(dates)
    #     print(days_df)
    result = pd.DataFrame(data={"dates": dates, "t": range(0, length)})
    result["weekday"] = pd.to_datetime(result["dates"]).apply(
        lambda x: x.strftime("%A")
    )
    return result



@ignore_warnings(category=ConvergenceWarning)
# this uses the combined dataset to find the GP mean and std
def model_prep_gp(groups_obs, slice_idx, kernel):
    #     print(slice_idx)
    comb = groups_obs.get_group(slice_idx).copy()
    comb["Date"] = pd.to_datetime(comb["Date"], format="%Y-%m-%d")
    t_df = get_dates([slice_idx[1]])
    t_df["dates"] = pd.to_datetime(t_df["dates"], format="%Y-%m-%d")

    # indices of the data
    x = t_df.t[t_df.dates.isin(comb.Date)].values
    x = x.reshape((-1, 1))

    # all the indices
    t = t_df.t.values.reshape((-1, 1))
    y = (
        comb[comb.Date.isin(t_df.dates)]
        .groupby("Date")
        .aggregate({"TotalFlow": "mean"})
    )
    #         y = (y - comb.TotalFlow.mean())/comb.TotalFlow.std()
    # y = y / comb.TotalFlow.mean()
    adj = y.TotalFlow.mean()
    # y = y - adj
    
    #     print(y)
    if y.isnull().values.any() > 0:
        print(slice_idx)

    if x.shape[0] != len(y):
        print(slice_idx)

    np.random.seed(1)
    # gpr = GaussianProcessRegressor(
    #     kernel=kernel, random_state=0, normalize_y=False
    # ).fit(x, y)
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True).fit(x, y)
    y_pred = gpr.predict(t, return_std=False)
    # y_pred_new = y_pred + adj
    y_pred_new = y_pred

    return t, y_pred_new

def get_reg_training_data(groups, slice_idx, t, y_pred_new):
    group_mean_obs_nn = y_pred_new.mean()
    group_std_obs_nn = y_pred_new.std()

    group_flow = groups.get_group(slice_idx).TotalFlow
    group_mean = group_flow.mean()
    group_std = group_flow.std()

    reg_training_dt = pd.DataFrame(
        {
            "mean_GP": [group_mean_obs_nn],
            "std_GP": [group_std_obs_nn],
            "mean": [group_mean],
            "std": [group_std],
        }
    )
    return reg_training_dt

def reg(df_GP):
    ########## mean 
    # defining feature matrix(X) and response vector(y)
    X = df_GP[["mean_GP", "std_GP"]]
    y1 = df_GP["mean"]
    # create linear regression object
    reg_mean = linear_model.LinearRegression()

    # train the model using the training sets
    reg_mean = reg_mean.fit(X, y1)
    
    mean_pred = reg_mean.predict(X)
    ######### std
    # defining feature matrix(X) and response vector(y)
    X = df_GP[["mean_GP", "std_GP"]]
    y2 = df_GP["std"]
    # create linear regression object
    reg_std = linear_model.LinearRegression()

    # train the model using the training sets
    reg_std = reg_std.fit(X, y2)
    std_pred = reg_std.predict(X)
    
    return reg_mean, reg_std, mean_pred, std_pred


def model_training(
    data,
    data_obs_new,
    data_obs,
    kernel,
    training_lst,
    # cutoff
):
    groups = data.groupby(["ID", "Weekday", "Hour"])
    groups_obs = data_obs.groupby(["ID", "Weekday", "Hour"])
    groups_new = data_obs_new.groupby(["ID", "Weekday", "Hour"]) # for GP only
    # group_lst = list(groups_new.groups)
    ############################################ GP ############################################
    df_GP = pd.DataFrame(columns=["mean_GP", "std_GP", "mean", "std"])
    p5=[]
    p25=[]
    p50=[]
    p75=[]
    p95=[]
    for i in tqdm(range(len(training_lst))):
        slice_idx = training_lst[i]
        t, y_pred_new = model_prep_gp(
                groups_obs=groups_new,
                slice_idx=slice_idx,
                kernel=kernel,
            )
        temp = get_reg_training_data(groups, slice_idx, t, y_pred_new) # combine the true mean and std with the GP mean and std
        df_GP = pd.concat([df_GP, temp])
        p5.append(np.quantile(y_pred_new,0.05))
        p25.append(np.quantile(y_pred_new,0.25))
        p50.append(np.quantile(y_pred_new,0.5))
        p75.append(np.quantile(y_pred_new,0.75))
        p95.append(np.quantile(y_pred_new,0.95))
    ############################################ regression model ############################################
    reg_mean, reg_std, mean_pred, std_pred = reg(df_GP)
    print(df_GP.shape)

    ############################################ training error ############################################
    logit_df = pd.DataFrame(
        {
            # "Date": pd.Series([], dtype="datetime64[ns]"),
            # "ID": pd.Series([], dtype="int64"),
            "TotalFlow": pd.Series([], dtype="float64"),
            "Anomaly": pd.Series([], dtype="bool"),
            "mean_pred": pd.Series([], dtype="float64"),
            "std_pred": pd.Series([], dtype="float64"),
        }
    )
    for i in tqdm(range(len(training_lst))):
        slice_idx = training_lst[i]
        # temp = groups_obs.get_group(slice_idx)[["Date", "ID", "Anomaly", "TotalFlow"]].copy()
        temp = groups_obs.get_group(slice_idx)[["TotalFlow","Anomaly"]].copy()
        #         print("mean", mean_pred)
        temp["mean_pred"] = mean_pred[i]
        temp["std_pred"] = std_pred[i]
        temp["p5"] = p5[i]
        temp["p25"] = p25[i]
        temp["p50"] = p50[i]
        temp["p75"] = p75[i]
        temp["p95"] = p95[i]
        logit_df = pd.concat([logit_df, temp])

    logit_df["TotalFlow_new"] = (
        abs(logit_df["TotalFlow"] - logit_df["mean_pred"]) / logit_df["std_pred"]
    )
    # logit_df["Anomaly_new"] = np.where(logit_df["Anomaly"] == True, 1, 0)

    logit_df["p5_new"] = (logit_df["p5"] - logit_df["mean_pred"])/logit_df["std_pred"]
    logit_df["p25_new"] = (logit_df["p25"] - logit_df["mean_pred"])/logit_df["std_pred"]
    logit_df["p50_new"] = (logit_df["p50"] - logit_df["mean_pred"])/logit_df["std_pred"]
    logit_df["p75_new"] = (logit_df["p75"] - logit_df["mean_pred"])/logit_df["std_pred"]
    logit_df["p95_new"] = (logit_df["p95"] - logit_df["mean_pred"])/logit_df["std_pred"]


    # define the training data 
    # X = np.array(logit_df.TotalFlow_new).reshape(-1, 1)
    # X = logit_df[["p5","p25","p50","p75","p95","TotalFlow_new"]]
    X = logit_df[["p5_new","p25_new","p50_new","p75_new","p95_new","TotalFlow_new"]]
    # print(X)
    # y = logit_df.Anomaly_new
    y = logit_df.Anomaly

    #### oversampling 
    # define oversampling strategy
    oversample = RandomOverSampler(sampling_strategy='minority')
    p = 0.4
    # oversample = RandomOverSampler(sampling_strategy=p)
    # fit and apply the transform
    X_over, y_over = oversample.fit_resample(X, y)

    # print(y.value_counts())        
    # print(y_over.value_counts())

    # #### logistic model
    # # fit the logit model
    # model = LogisticRegression(solver="liblinear", random_state=0)
    # model.fit(X, y)
    # # model.fit(X_over, y_over)
    # # logit_df["Anomaly_prob"] = [prob[1] for prob in model.predict_proba(np.array(logit_df.TotalFlow_new).reshape(-1, 1))]
    # logit_df["Anomaly_prob"] = [prob[1] for prob in model.predict_proba(X)]
    
    # optimal_F1 = 0
    # optimal_cutoff = cutoff[0]
    # for cut_prob in cutoff:
    #     logit_df["Anomaly_pred"] = np.where(
    #         logit_df.Anomaly_prob > cut_prob, True, False
    #     )
    #     # compute the F1 score
    #     curr_F1 = f1_score(list(logit_df.Anomaly), list(logit_df.Anomaly_pred))
    #     #         print(pd.crosstab(logit_df.Anomaly, logit_df.Anomaly_pred))
    #     if optimal_F1 < curr_F1:
    #         optimal_F1 = curr_F1
    #         optimal_cutoff = cut_prob

    # print("cutoff: ", optimal_cutoff, "training F1: ", optimal_F1)

    
    # Create parameters to search
    params = {
        'learning_rate': [0.03, 0.05],
        'n_estimators': [1000,1500],
        'max_depth':[8,10],
        'reg_alpha':[0.3,0.5],
        'subsample':[0.5,1]
        }
    # Initializing the XGBoost Regressor
    xgb_model = xgb.XGBClassifier()
    # Gridsearch initializaation
    gsearch = GridSearchCV(xgb_model, params,
                        verbose=False,
                        cv=5,
                        n_jobs=2)
    
    gsearch.fit(X, y)
    #Printing the best chosen params
    # print("Best Parameters :",gsearch.best_params_)
    params = {'objective':'binary:logistic', 'booster':'gbtree'}
    # Updating the parameter as per grid search
    params.update(gsearch.best_params_)
    
    # Initializing the XGBoost Regressor
    model = xgb.XGBClassifier(**params)
    print(model.get_params())
    
    # # Fitting model on tuned parameters
    model.fit(X, y)
    y_predict = model.predict(X)

    optimal_F1 = f1_score(list(y), list(y_predict))
    return {
        "reg_mean": reg_mean,
        "reg_std": reg_std,
        "logit_model": model,
        "training_F1": optimal_F1,
        # "cutoff": optimal_cutoff,
    }


def model_pred_logit(reg_mean, reg_std, logit_model, groups_obs_new, groups_obs, slice_idx, kernel,unknowm_anomaly):
    # find the pooled data from nn
    _, y_pred_new= model_prep_gp(groups_obs_new,slice_idx, kernel)
    # print(y_pred_new)
    mean_GP = y_pred_new.mean()
    std_GP = y_pred_new.std()
    X_test = np.array([[mean_GP, std_GP]])
    group_obs = groups_obs.get_group(slice_idx)
#     print(group)
    # for validation, true anomaly status is known
    if unknowm_anomaly == False:
        anomaly_pred = group_obs[["ID","Fraction_Observed","TotalFlow","Anomaly"]].copy()
    # for city 5, true status is unknowm, all columns are retained
    else:
        anomaly_pred = group_obs[["ID","Fraction_Observed","TotalFlow"]].copy()
      
    # predict the mean and std
    anomaly_pred['mean'] = reg_mean.predict(X_test)[0]
    anomaly_pred['std'] = reg_std.predict(X_test)[0]

    # compute the quantiles from GP prediction
    anomaly_pred['p5'] = np.quantile(y_pred_new,0.05)
    anomaly_pred['p25'] = np.quantile(y_pred_new,0.25)
    anomaly_pred['p50'] = np.quantile(y_pred_new,0.5)
    # anomaly_pred['median'] = np.median(y_pred_new)
    anomaly_pred['p75'] = np.quantile(y_pred_new,0.75)
    anomaly_pred['p95'] = np.quantile(y_pred_new,0.95)

    # plt.plot(y_pred_new)

    anomaly_pred["p5_new"] = (anomaly_pred["p5"] - anomaly_pred["mean"])/anomaly_pred["std"]
    anomaly_pred["p25_new"] = (anomaly_pred["p25"] - anomaly_pred["mean"])/anomaly_pred["std"]
    anomaly_pred["p50_new"] = (anomaly_pred["p50"] - anomaly_pred["mean"])/anomaly_pred["std"]
    anomaly_pred["p75_new"] = (anomaly_pred["p75"] - anomaly_pred["mean"])/anomaly_pred["std"]
    anomaly_pred["p95_new"] = (anomaly_pred["p95"] - anomaly_pred["mean"])/anomaly_pred["std"]

    # compute the normalized totalflow
    anomaly_pred["TotalFlow_new"] = (
    abs(anomaly_pred["TotalFlow"] - anomaly_pred['mean']) / anomaly_pred['std']
)
    # predict the probability from the logit model
    # anomaly_pred["Anomaly_prob"] = [prob[1] for prob in logit_model.predict_proba(np.array(anomaly_pred.TotalFlow_new).reshape(-1, 1))]
    # Xtest = anomaly_pred[["p5","p25","p50","p75","p95","TotalFlow_new"]]
    Xtest = anomaly_pred[["p5_new","p25_new","p50_new","p75_new","p95_new","TotalFlow_new"]]
    
    ### logistic regression
    # anomaly_pred["Anomaly_prob"] = [prob[1] for prob in logit_model.predict_proba(Xtest)]
    
    #### SVM model
    anomaly_pred["Anomaly_pred"] = logit_model.predict(Xtest)
    return anomaly_pred

def model_prediction(
    data_obs_new,
    data_obs,
    model,
    kernel,
    validation_lst,
    unknowm_anomaly,
):
    groups_obs_new = data_obs_new.groupby(["ID", "Weekday", "Hour"])
    groups_obs = data_obs.groupby(["ID", "Weekday", "Hour"])
    # group_lst = list(groups_obs.groups)
    #### model prediction
    logit_df = pd.DataFrame(
        {
#             "Timestamp": pd.Series([], dtype="datetime64[ns]"),
            "ID": pd.Series([], dtype="int64"),
            "TotalFlow": pd.Series([], dtype="float64"),
            "Anomaly": pd.Series([], dtype="bool"),
        }
    )
    for j in tqdm(range(len(validation_lst))):
        slice_idx = validation_lst[j]
        temp = model_pred_logit(
            reg_mean=model["reg_mean"],
            reg_std=model["reg_std"],
            logit_model=model["logit_model"],
            groups_obs_new=groups_obs_new,
            groups_obs=groups_obs,
            slice_idx=slice_idx,
            kernel=kernel,
            unknowm_anomaly=unknowm_anomaly,
        )

        logit_df = pd.concat([logit_df, temp])

    # ### logistic regression
    # # use the optimal cutoff to compute the validation F1
    # # validation F1 score
    # logit_df["Anomaly_pred"] = np.where(
    #     logit_df.Anomaly_prob > model["cutoff"], True, False
    # )

    # test_F1 = f1_score(list(logit_df.Anomaly), list(logit_df.Anomaly_pred))

    ### SVM
    test_F1 = f1_score(list(logit_df.Anomaly), list(logit_df.Anomaly_pred))

    return {"validation_F1":test_F1}

def get_fraction_slice(fraction_groups, fraction):
    fraction_groups_lst = list(fraction_groups.groups)
    res = []
    for i in fraction_groups_lst:
        if i[3] == fraction:
            res.append(i[0:3])
    return res

def train_val_test(seed, train_size, validation_size, slice_list):
    # sample the training, validation, and test dataset
    random.seed(seed)
    training = random.sample(slice_list, train_size)
    temp = [elem for elem in slice_list if not elem in training]
    validation = random.sample(temp, validation_size)
    # test = random.sample(temp2, test_size)
    return training, validation

 
def run_train_fraction2(fraction_ind, data, data_obs_new, data_obs, kernel, sampled_IDs, p_training, p_validation):
    fraction_groups = data_obs[data_obs.ID.isin(sampled_IDs)].groupby(
        ["ID", "Weekday", "Hour", "Fraction_Observed"]
    )
    slice_p = get_fraction_slice(fraction_groups, fraction_ind)
    training_lst, validation_lst = train_val_test(
        seed=123,
        train_size=math.floor(len(slice_p) * p_training),
        validation_size=math.floor(len(slice_p) * p_validation),
        slice_list=slice_p
    )
    print(math.floor(len(slice_p) * p_training))
    model = model_training(
        data=data,
        data_obs_new=data_obs_new,
        data_obs=data_obs,
        kernel=kernel,
        training_lst=training_lst,
        # cutoff=cutoff
    )
    print("This is a test print of training for slice", fraction_ind,end='\r')
    validation = model_prediction(
        data_obs_new=data_obs_new,
        data_obs=data_obs,
        model=model,
        kernel=kernel,
        validation_lst=validation_lst,
        unknowm_anomaly=False,
    )
    print("This is a test print of validation for slice", fraction_ind,end='\r')
    # return {"model": model, "val_df": validation}
    return {"reg_mean": model['reg_mean'],
        "reg_std": model['reg_std'],
        "logit_model": model['logit_model'],
        "training_F1": model['training_F1'],
        # "cutoff": model['cutoff'],
        # "pred_df":validation['pred_df'],
        "validation_F1":validation['validation_F1']}