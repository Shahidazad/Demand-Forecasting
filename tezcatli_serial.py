#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from tezcatli_scripts import load_data, utils, pre_process , write_to_database as w2d #, get_ts_features , fit_models,
from tezcatli_scripts.fit_models import  Darts, Orbit

### For tracking execution times
from os import path
import time
#import random
from random import getrandbits, seed
import pickle

#For reading options from command line
import sys, getopt

import warnings
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [20,6]
pd.set_option('display.max_columns',12)
pd.set_option('display.width', 1000)


# In[2]:


from darts.models import Prophet
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    NaiveMean,
    ExponentialSmoothing,
    AutoARIMA,
    #StandardRegressionModel,
    Theta,
    FFT,
    Croston,
    LightGBMModel,
    RandomForest,
    RegressionEnsembleModel,
    TBATS,
    BATS,
    RegressionModel)
from darts.metrics import mape, smape, mase
from orbit.models import DLT


# In[3]:


#%% Models to run ###
model_frames = ['Darts','Orbit']
dart_models = [ExponentialSmoothing(), NaiveSeasonal(), NaiveDrift(), NaiveMean(), AutoARIMA(), Theta(), FFT(), Prophet(),Croston(),LightGBMModel(lags=1),RandomForest(lags= 1,random_state=2309),RegressionEnsembleModel(forecasting_models=[ExponentialSmoothing(), NaiveSeasonal(),AutoARIMA(),TBATS()],regression_train_n_points=24),TBATS(),BATS(),RegressionModel(lags=1),RegressionModel(lags=10)]#, StandardRegressionModel()]
dart_models_names = ['ExponentialSmoothing', 'NaiveSeasonal','NaiveDrift','NaiveMean','AutoARIMA','Theta', 'FFT','Prophet','Croston','LightGBMModel','RandomForest','RegressionEnsembleModel','TBATS','BATS','RegressionModelL1','RegressionModelL10']#, 'StandardRegression']
models = [DLT(),ExponentialSmoothing(), NaiveSeasonal(), NaiveDrift(), NaiveMean(), AutoARIMA()]#, Theta(), FFT(), Prophet(),Croston(),LightGBMModel(lags=1),RandomForest(lags= 1,random_state=2309),RegressionEnsembleModel(forecasting_models=[ExponentialSmoothing(), NaiveSeasonal(),AutoARIMA(),TBATS()],regression_train_n_points=24),TBATS(),BATS(),RegressionModel(lags=1),RegressionModel(lags=10)]#, StandardRegressionModel()]
models_names = ['Orbit','ExponentialSmoothing', 'NaiveSeasonal','NaiveDrift','NaiveMean','AutoARIMA']#,'Theta', 'FFT','Prophet','Croston','LightGBMModel','RandomForest','RegressionEnsembleModel','TBATS','BATS','RegressionModelL1','RegressionModelL10']#, 'StandardRegression']
orbit_models = [DLT()]
orbit_models_names = ['DampedLinearTrend']


# In[4]:


run_config = utils.read_params_in_from_json('run_config.json')
group_key = run_config['dimensions'].split('-')
#seed(2309)
run_id = getrandbits(32)
run_date = dt.datetime.today().date()
run_datascientist = run_config['data_scientist']
run_scope = run_config['scope']
run_response = run_config['response']
run_timegrain = run_config['timegrain']
run_dimensions = run_config['dimensions']
run_type = run_config['type']
holdout_horizon = run_config['train_horizon']
forecast_horizon = run_config['forecast_horizon']

test_run = True

write_file = 'yes'


# In[5]:


prep_comp_prod = pd.read_pickle('prep_init.pkl')
prep_comp_prod.head()
print('prep_comp_prod',prep_comp_prod)

# In[6]:


t_train_start = time.time()
prod_dfs, prod_accs,failed_keys = [],[],[]
#ts_feats = []
## Certain stat models have constraints on length of time series , see later checks##
time_models = ['ExponentialSmoothing']
time_models2 = ['AutoARIMA']
time_models3 = ['RegressionEnsembleModel','TBATS','BATS']
cnt = 0
keys = prep_comp_prod['group_key'].unique()


# In[7]:


#key = np.where(keys=='Midwest Central_C+ St Plk')
#keys[key].item()
key = 'Midwest Central_C+ St Plk' 


# In[8]:


prod_df = prep_comp_prod[prep_comp_prod['group_key']==key]
prod_df


# In[9]:


models_list = list(zip(models,models_names))
model_dfs, model_accs = [],[]


# In[10]:


models_list


# In[11]:


run_mofcst = dt.datetime(run_config['current_year'],run_config['current_month'],1)
train_date = run_mofcst + relativedelta(months=-holdout_horizon)


# In[12]:


log_file_path = path.join(path.dirname(path.abspath('params/log.conf')), 'log.conf')
jhds_logger = utils.setup_logger(log_file_path)
jhds_logger.info('Finished setup')


# In[13]:

# print(prod_df.columns)
start_time = time.perf_counter()
print('prod_df_final',prod_df)
print('train_date',train_date)
print('run_mofcst',run_mofcst)
for model,name in list(zip(models,models_names)):
    #### Instantiate model framework
    if (name == 'Orbit'):
        model_frame = Orbit(model,prod_df,run_mofcst,train_date=train_date,forecast_horizon=None)
    else:
        model_frame = Darts(model,prod_df, run_mofcst, train_date,forecast_horizon=None)
    #print(model_frame)

    #### Create time series
    #prod_ts = pre_process.create_ts(prod_df,run_mofcst)
    model_frame.prep_data()

    #### Split train and test sets for holdout accuracy
    #train,val = prod_ts.split_before(pd.Timestamp(train_date))
    model_frame.split_data()
    #TODO create fit function (returns model params)
    # check for length , Expo can't handle less than 24, autoarima needs 30
    if ( (len(model_frame.train)<24) & (name in time_models) ):
        failed_keys.append((key,name))
        continue
    elif ( (len(model_frame.train)<30) & (name in time_models2)) :
        failed_keys.append((key,name))
        continue
    elif ( (name in time_models3) & (model_frame.get_train_df_ordervol().tail(12).sum()<12) ):
        failed_keys.append((key,name))
        continue
    #### Fit and pred ##
    t_modelfitpred_start = time.time()
    try:
        model_frame.train_model()
        #model.fit(train)
    except ZeroDivisionError:
        jhds_logger.error(f'A zero division error occurred with key {key} in training model {name}')

    try:
        pred = model_frame.pred_model()
        #pred = model.predict(len(val))
    except ValueError:
        jhds_logger.error(f'Training model {name} with key {key} failed due to NaN, infinity or too large number')

    pred_df = pred.pd_dataframe()
    pred_df['model'] = name
    pred_df.rename(columns={'0':'fcst'},inplace=True)
    model_dfs.append(pred_df)
    #### Accuracies ##
    accuracies = pd.DataFrame()
    accuracies['mape'] = pd.Series(mape(model_frame.val,pred))
    accuracies['smape'] = pd.Series(smape(model_frame.val,pred))
    accuracies['mase'] = pd.Series(mase(model_frame.val,pred,insample=model_frame.train))
    accuracies['model'] = name
    model_accs.append(accuracies)
    t_modelfitpred_end = time.time()
    jhds_logger.info(f'Finished fitting model {name}, it took {"{:.2f}".format((t_modelfitpred_end-t_modelfitpred_start)/60)} minutes')
finish_time = time.perf_counter()
print(f"Program finished in {finish_time-start_time} seconds")
print(model_dfs, model_accs)




