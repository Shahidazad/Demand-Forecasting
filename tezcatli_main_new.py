# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:51:05 2022

@author: PabloT
"""
#%% Import general libraries
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

t_script_start = time.time()
#%% Import time series libraries
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

#%% Setup Run ###
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

#%% Models to run ###
model_frames = ['Darts','Orbit']
dart_models = [ExponentialSmoothing(), NaiveSeasonal(), NaiveDrift(), NaiveMean(), AutoARIMA(), Theta(), FFT(), Prophet(),Croston(),LightGBMModel(lags=1),RandomForest(lags= 1,random_state=2309),RegressionEnsembleModel(forecasting_models=[ExponentialSmoothing(), NaiveSeasonal(),AutoARIMA(),TBATS()],regression_train_n_points=24),TBATS(),BATS(),RegressionModel(lags=1),RegressionModel(lags=10)]#, StandardRegressionModel()]
dart_models_names = ['ExponentialSmoothing', 'NaiveSeasonal','NaiveDrift','NaiveMean','AutoARIMA','Theta', 'FFT','Prophet','Croston','LightGBMModel','RandomForest','RegressionEnsembleModel','TBATS','BATS','RegressionModelL1','RegressionModelL10']#, 'StandardRegression']
models = [DLT()]#,ExponentialSmoothing(), NaiveSeasonal(), NaiveDrift(), NaiveMean(), AutoARIMA(), Theta(), FFT(), Prophet(),Croston(),LightGBMModel(lags=1),RandomForest(lags= 1,random_state=2309),RegressionEnsembleModel(forecasting_models=[ExponentialSmoothing(), NaiveSeasonal(),AutoARIMA(),TBATS()],regression_train_n_points=24),TBATS(),BATS(),RegressionModel(lags=1),RegressionModel(lags=10)]#, StandardRegressionModel()]
models_names = ['Orbit']#,'ExponentialSmoothing', 'NaiveSeasonal','NaiveDrift','NaiveMean','AutoARIMA','Theta', 'FFT','Prophet','Croston','LightGBMModel','RandomForest','RegressionEnsembleModel','TBATS','BATS','RegressionModelL1','RegressionModelL10']#, 'StandardRegression']
orbit_models = [DLT()]
orbit_models_names = ['DampedLinearTrend']
#%% Main Method
def main():
    #%%% Disable warnings
    warnings.filterwarnings("ignore")
    #%%% Set dates
    ## Set the date of the month to be run (one more than what data you have)
    run_mofcst = dt.datetime(run_config['current_year'],run_config['current_month'],1)
    train_date = run_mofcst + relativedelta(months=-holdout_horizon)
    order_min_date = dt.datetime(2014,4, 1).date()
    order_max_date = run_mofcst + relativedelta(months=-1)
    
    #%% Setting up logging ###
    log_file_path = path.join(path.dirname(path.abspath('params/log.conf')), 'log.conf')
    jhds_logger = utils.setup_logger(log_file_path)
    jhds_logger.info('Finished setup')
    
    #%% Load data ###
    #data = pd.read_csv('data/huitzilo_orders_data.csv')
    #%%%% First write most recent data to db
    t_loaddata_start = time.time()
    data = pd.read_parquet('data/tezcatli_orders_data.parquet')
    #data = load_data.load_from_db()
    
    if not test_run:
        orders = load_data.load_new_data()
        #w2d.write_new_data_db(orders)
        data = pd.concat([data,orders],ignore_index=True)
        data.to_parquet('data/tezcatli_orders_data.parquet')
    
    # data = data[data['Region Name']=='Texas and South Plains']
    # data = data[data['District Name']=='South Plains']
    # data = data[data['Product Line']=='Trim']
    # data = data[data['Product Segment']=='Exterior']
    ### Use only specified forecast groups ###
    top_fcst_grps = ["Pr Plk","Pr HLD","Pr Pnl","C+ St Plk","Int 1/2 Inch","C+ St NT3","Int 1/4 Inch","C+ St Pnl","Pr CemPre","Pr Soff 12'","C+ St HLD","Pr Soff 8'"]
    data = data[data['forecast_group'].isin(top_fcst_grps)]
    t_loaddata_end = time.time()
    jhds_logger.info(f'Finished loading data, it took {"{:.2f}".format((t_loaddata_end-t_loaddata_start)/60)} minutes')
    #### Finished loading data ###

    #%%% Read parameters from CLI if any
    try:
        opts, args = getopt.getopt(sys.argv[1:],'r:d:s:mo:m:w:')
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    write_file = 'yes'
    for opt, a in opts:
        if opt == '-r':
            try:
                assert a in data['Region Name'].unique()
                data = data[data['Region Name']==a]
            except:
                print('Name of region is incorrect')
                exit()
        elif opt == '-d':
            try:
                assert a in data['District Name'].unique()
                data = data[data['District Name']==a]
            except:
                print('Name of district is incorrect')
                exit()
        elif opt == '-s':
            try:
                assert a in data['Product Segment'].unique()
                data = data[data['Product Segment']==a]
            except:
                print('Product Segment is incorrect')
                exit()
        elif opt == '-w':
            write_file = a
        else:
            assert False, 'unhandled option'

    #%%% Prep data ###
    t_prepdata_start = time.time()
    data['Order Create Date']=pd.to_datetime(data['Order Create Date'])
    # data['Order Create Date2']=data['Order Create Date'].to_numpy().astype('datetime64[M]')
    # data['Order Create Date2']=data['Order Create Date'].dt.to_period('M').dt.to_timestamp()


    ### Prep the data - drop data exports and unused product segments
    data = data[data['Region Name']!='Export/Interco']
    data = data[data['Product Segment'].isin(['Exterior','Interior'])]

    ### check unique value --- 4684
    #len(data[['District Name', 'Region Name', 'Product Segment', 'Product Line', 'Product Family', 'Product Group', 'Product Finish', 'Color Type']].drop_duplicates())

    ### Change columns names
    data.columns = data.columns.str.replace(' - ', '_')
    data.columns = data.columns.str.replace(' ', '_')

    ### Replace nan value
    data[group_key] = data[group_key].fillna('NA')
    #data.replace(np.nan,'NA', inplace=True)

    ### Create grouping key
    data['group_key'] = data[group_key].agg('_'.join,axis=1)

    ### Replace original 0s with NAs
    data.dropna(subset=['Order_Volume_(STD)'],inplace=True)
    #data = data[data["Order_Volume_(STD)"]!=0]
    #data['Order_Volume_(STD)'] = np.where(data['Order_Volume_(STD)']==0,np.NAN,data['Order_Volume_(STD)'])

    ### Group by month and group key
    data = data.set_index('Order_Create_Date').groupby([pd.Grouper(freq='MS'),'group_key'])['Order_Volume_(STD)'].sum().reset_index()
    #data = data.groupby(['Order_Create_Date','group_key'])['Order_Volume_(STD)'].sum().reset_index()

    ### Filter for orders until month of forecast
    data = data[data['Order_Create_Date']<=run_mofcst]

    ### Create keys for date indexing
    allkeys = data['group_key'].unique()

    ### Fix the dates
    idx = pd.date_range(order_min_date,order_max_date,freq='MS',name='Order_Create_Date')
    data = pre_process.fix_dates(data,idx,allkeys)

    ### Get counts of values higher than 0 ###
    data_counts = data[data['Order_Volume_(STD)']>0.000001][['group_key']].value_counts().reset_index(name='counts')
    ### Find complete and incomplete data
    incomplete_data_df = data_counts.query('counts<=6')
    #incomplete_data_df = data.iloc[incomplete_data_idx]

    complete_data_df = data_counts.query('counts>6')
    #complete_data_df = data.iloc[complete_data_idx]

    ### Change date type of order volume to int
    # data['Order_Volume_(STD)'] = data['Order_Volume_(STD)'].fillna(0.000001)
    # #data['Order_Volume_(STD)'] = np.where(data['Order_Volume_(STD)']=='NA','0.000001',data['Order_Volume_(STD)'])
    # #data = data[data["Order_Volume_(STD)"].apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
    # data = data[data["Order_Volume_(STD)"].apply(lambda x: type(x) in [ float, np.float64])]

    ### Remove the rows have NA on Order_Volume_(SQFT)
    #data = data[data["Order_Volume_(STD)"]!=0]

    ### Filter for top or complete products ###
    #complete_data_df = data[['group_key']].value_counts().reset_index(name='counts').query('counts>6')
    prep_init_prod = data.merge(complete_data_df[['group_key']].drop_duplicates(),how='inner',on='group_key')
    prep_init_prod['run_id'] = run_id

    ### Record incomplete-data products
    #incomplete_data_df = data[['group_key']].value_counts().reset_index(name='counts').query('counts<=6')
    prep_incomplete_prod = data.merge(incomplete_data_df[['group_key']].drop_duplicates(),how='inner',on='group_key')
    prep_incomplete_prod['run_id'] = run_id
    prep_incomplete_prod['fail_point'] = 'insufficient data'


    ### Create keys for date indexing
    keys = prep_init_prod['group_key'].unique()
    
    ### Group by date and group key
    #prep_init_prod = prep_init_prod.groupby(['Order_Create_Date','group_key'])['Order_Volume_(SQFT)'].sum().reset_index()

    #%% Pickle the prepped data
    prep_init_file = 'prep_init.pkl'
    with open(prep_init_file,'wb') as f:
        pickle.dump(prep_init_prod,f)

    #%% Dataframe for modeling
    prep_comp_prod = prep_init_prod.copy()

    t_prepdata_end = time.time()
    jhds_logger.info(f'Finished prepping data, it took {"{:.2f}".format((t_prepdata_end-t_prepdata_start)/60)} minutes')

    #%% Fit best model ###
    ### Cycle through groups ###

    t_train_start = time.time()
    prod_dfs, prod_accs,failed_keys = [],[],[]
    #ts_feats = []
    ## Certain stat models have constraints on length of time series , see later checks##
    time_models = ['ExponentialSmoothing']
    time_models2 = ['AutoARIMA']
    time_models3 = ['RegressionEnsembleModel','TBATS','BATS']
    cnt = 0

    ## Test
    #keys = ['Texas and South Plains_Exterior']
    # prep_comp_prod['Order_Volume_(STD)'] = prep_comp_prod['Order_Volume_(STD)']/1e6

    for key in keys :
        cnt +=1
        len_keys = len(keys)
        jhds_logger.info(f'Working on item : {key}, key {cnt} out of {len_keys} ')
        prod_df = prep_comp_prod[prep_comp_prod['group_key']==key]

        #### Cycle through models ###
        t_model_start = time.time()
        #jhds_logger.info(f'Cycling through models for {key} ')
        model_dfs, model_accs = [],[]
        seed(2309)
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

        accs_df = pd.concat(model_accs)
        accs_df['group_key'] = key
        models_df = pd.concat(model_dfs)
        models_df['group_key'] = key
        prod_dfs.append(models_df)
        prod_accs.append(accs_df)
        #t_inner.toc('Product Model fit and preds took ')
        t_model_end = time.time()
        jhds_logger.info(f'Finished cycling through models for key: {key}, cycle took {"{:.2f}".format((t_model_end-t_model_start)/60)} minutes')

    accs_list_file = 'accs_list.pkl'
    preds_list_file = 'preds_list.pkl'
    with open(accs_list_file,'wb') as f:
        pickle.dump(prod_accs,f)
    with open(preds_list_file,'wb') as f:
        pickle.dump(prod_dfs,f)
    jhds_logger.info('Pickled accuracies and predictions in holdout')
    prods_mods_df = pd.concat(prod_dfs).reset_index()
    prods_accs_df = pd.concat(prod_accs)

    ## Failed logging ##
    failed_df = pd.DataFrame({'group_key':failed_keys})

    ###
    holdout_accs = prods_accs_df.set_index(['model','group_key']).reset_index()
    failed_accuracy = holdout_accs[holdout_accs['smape'].isnull()][['group_key','model']]
    failed_accuracy['fail_point'] = 'accuracy_calc'
    failed_df = pd.concat([failed_df,failed_accuracy])
    holdout_accs = holdout_accs.dropna(subset=['smape'])

    #%% Get best model for every product ###
    holdout_winners = holdout_accs.loc[holdout_accs.groupby(['group_key'])['mase'].idxmin()]
    winners = dict(zip(holdout_winners['group_key'],holdout_winners['model']))

    ### Get holdout df

    t_train_end = time.time()
    jhds_logger.info(f'Finished training all keys , cycle took {"{:.2f}".format((t_train_end-t_train_start)/60)} minutes')
    #%% Generate forecasts for every key with best model ###

    ### Cycle through products ###
    forecasts = []
    cnt = 0
    t_fcst_start = time.time()
    jhds_logger.info('Start of forecasting')

    for key in winners.keys():
        t_modelfcst_start = time.time()
        cnt +=1
        len_keys = len(winners.keys())
    ### Fit , predict into the future
        prod_df = prep_comp_prod[prep_comp_prod['group_key']==key]

        wmodel_name = winners.get(key)
        wmodel = models[models_names.index(wmodel_name)]

        #### Instantiate model framework
        if (wmodel_name == 'Orbit'):
            wmodel_frame = Orbit(wmodel,prod_df,run_mofcst,train_date=train_date, forecast_horizon=forecast_horizon)
        else:
            wmodel_frame = Darts(wmodel,prod_df, run_mofcst, train_date,forecast_horizon=forecast_horizon)

        #### Create time series ##
        #prod_ts = pre_process.create_ts(prod_df,run_mofcst)
        wmodel_frame.prep_data()

        #fcst_horizon = forecast_horizon

        #### Fit and predict future
        try:
            #wmodel_frame.fit(prod_ts)
            wmodel_frame.train_model(train_mode=False)
        except Exception as e:
            jhds_logger.exception(f'Exception {e} occurred in key {key} and model {wmodel_name}')

        try:
            fcst = wmodel_frame.pred_model(train_mode=False)
        except ValueError:
            jhds_logger.error(f'Forecasting key {key} failed due to NaN, infinity or too large number')

        fcst_df = fcst.pd_dataframe()
        fcst_df['model'] = wmodel_name
        fcst_df['group_key'] = key
        fcst_df.rename(columns={'0':'fcst'},inplace=True)
        forecasts.append(fcst_df)
        #t_inner.toc(f'Product Model fit and preds for {key} took ')
        t_modelfcst_end = time.time()
        jhds_logger.info(f'Finished forecasting for {key}, it took {"{:.2f}".format((t_modelfcst_end-t_modelfcst_start)/60)} minutes')

    fcst_list_file = 'fcst_list.pkl'
    with open(fcst_list_file,'wb') as f:
        pickle.dump(forecasts,f)
    t_fcst_end = time.time()
    jhds_logger.info(f'End of Forecasting: run took {"{:.2f}".format((t_fcst_end-t_fcst_start)/60)} minutes')

    forecast_df = pd.concat(forecasts)
    forecast_df.reset_index(inplace=True)
    forecast_df['Order_Create_Date'] = forecast_df['Order_Create_Date'].dt.date
    ### Create a lag ###
    forecast_df['lag'] = forecast_df.groupby('group_key')['Order_Create_Date'].cumcount()
    ### Add run id ###
    forecast_df['run_id'] = run_id
    ### Add series class ###
    forecast_df['series_class'] = 'ok'

    #%% Post process forecasts , ensure quality ###
    #### Check for negative forecasts and turn them to 0
    forecast_df['Order_Volume_(STD)'] = np.where(forecast_df['Order_Volume_(STD)']<0,0,forecast_df['Order_Volume_(STD)'])
    df1=forecast_df[['Order_Create_Date', 'Order_Volume_(STD)', 'model', 'group_key', 'series_class','run_id','lag']]

    prep_comp_prod['Order_Create_Date'] = prep_comp_prod['Order_Create_Date'].dt.date
    prep_comp_prod[group_key] = prep_comp_prod['group_key'].str.split('_',expand=True,n=len(group_key))

    #### insufficient data
    if len(prep_incomplete_prod)>0:
        ########### Filter for values > 0.000001 ##########
        prep_incomplete_prod_nonzero = prep_incomplete_prod[prep_incomplete_prod['Order_Volume_(STD)']>0.000005]
        prep_incomplete_prod_maxdate = prep_incomplete_prod_nonzero.sort_values('Order_Create_Date').groupby('group_key').tail(1)

        ######## no values in last 6 months
        prep_incomplete_prod0 = prep_incomplete_prod_maxdate[prep_incomplete_prod_maxdate['Order_Create_Date']<pd.Timestamp(train_date)]
        inc_keys = prep_incomplete_prod0['group_key'].unique()
        inc_fcst_idx = pd.date_range(run_mofcst,order_max_date + relativedelta(months=forecast_horizon),freq='MS',name='Order_Create_Date')#'cyear_month')
        ####### Check that the length of inc_keys is > 0 otherwise create empty data frame ########
        if len(inc_keys)>0 :
            incompletes_fcst = pre_process.create_incomplete_fcst(inc_fcst_idx, inc_keys, run_id)
            incompletes_fcst['lag'] = incompletes_fcst.groupby('group_key')['Order_Create_Date'].cumcount()
        else:
            incompletes_fcst = pre_process.create_incomplete_fcst(inc_fcst_idx, ['no_keys'], run_id)
            incompletes_fcst['lag'] = incompletes_fcst.groupby('group_key')['Order_Create_Date'].cumcount()

        ###### has values in last 6 months
        prep_incomplete_prod_maxdate6 = prep_incomplete_prod_maxdate[prep_incomplete_prod_maxdate['Order_Create_Date']>=pd.Timestamp(train_date)]
        prep_incomplete_prod6 = prep_incomplete_prod[prep_incomplete_prod['group_key'].isin(prep_incomplete_prod_maxdate6['group_key'])]
        prep_incomplete_prod6_forecast = prep_incomplete_prod6.groupby(['group_key'])['Order_Volume_(STD)'].agg('sum')/run_config['train_horizon']
        prep_incomplete_prod6_forecast = prep_incomplete_prod6_forecast.reset_index()
        prep_incomplete_prod6_forecast['forc_date'] = pd.to_datetime(run_mofcst)
        #forc_date = {"forc_date":pd.date_range(order_min_date,order_max_date,freq='MS') }
        forc_date = {"forc_date":inc_fcst_idx }
        prep_incomplete_prod6_forecast_all = prep_incomplete_prod6_forecast.complete(forc_date, by='group_key')
        #prep_incomplete_prod6_forecast_all = prep_incomplete_prod6_forecast.complete([forc_date], by='group_key')
        #prep_incomplete_prod6_forecast_all = prep_incomplete_prod6_forecast.complete([inc_fcst_idx], by='group_key')
        prep_incomplete_prod6_forecast_all['Order_Volume_(STD)'] = prep_incomplete_prod6_forecast_all['Order_Volume_(STD)'].fillna(prep_incomplete_prod6_forecast_all.groupby('group_key')['Order_Volume_(STD)'].transform('max'))
        prep_incomplete_prod6_forecast_all['model']='average values last 6 months'
        prep_incomplete_prod6_forecast_all['series_class']='insufficient data - values last 6 months'
        #prep_incomplete_prod6_forecast_all['lag'] = prep_incomplete_prod6_forecast_all.groupby('group_key')['Order_Create_Date'].cumcount()
        if len(prep_incomplete_prod6_forecast_all)>0 :
            prep_incomplete_prod6_forecast_all['lag'] = prep_incomplete_prod6_forecast_all.groupby('group_key')['series_class'].cumcount()
            #prep_incomplete_prod6_forecast_all['lag'] = prep_incomplete_prod6_forecast_all.groupby('group_key')['level_1'].cumcount()
        else:
            prep_incomplete_prod6_forecast_all = pre_process.create_incomplete_fcst(inc_fcst_idx, ['no_keys'], run_id)
            prep_incomplete_prod6_forecast_all['lag'] = prep_incomplete_prod6_forecast_all.groupby('group_key')['cyear_month'].cumcount()
        prep_incomplete_prod6_forecast_all['run_id'] = run_id


        inc_act_idx = pd.date_range(order_min_date,order_max_date,freq='MS',name='Order_Create_Date')
        ####### Check that the length of inc_keys is > 0 otherwise create empty data frame ########
        if len(inc_keys)>0 :
            incompletes_act = pre_process.fix_dates(prep_incomplete_prod,inc_act_idx,inc_keys)
            incompletes_act[group_key] = incompletes_act['group_key'].str.split('_',expand=True,n=len(group_key))
        else:
            incompletes_act = prep_incomplete_prod
            incompletes_act = incompletes_act.reindex(columns = incompletes_act.columns.tolist()+group_key)
        incompletes_act['series_class'] = 'insufficient data'
        incompletes_act['run_id'] = run_id
        ### Merge with prep_incomplete_prod to get all sku info and counts


        ### Incomplete actuals with values in 6 mo ###
        inc_keys6 = prep_incomplete_prod6['group_key'].unique()
        ####### Check that the length of inc_keys6 is > 0 otherwise create empty data frame ########
        if len(inc_keys6)>0:
            incompletes_act6 = pre_process.fix_dates(prep_incomplete_prod6, inc_act_idx, inc_keys6)
            incompletes_act6[group_key] = incompletes_act6['group_key'].str.split('_',expand=True,n=len(group_key))
        else:
            incompletes_act6 = prep_incomplete_prod6
            incompletes_act6 = incompletes_act6.reindex(columns = incompletes_act6.columns.tolist()+group_key)
        incompletes_act6['series_class'] = 'insufficient data - values last 6 months'
        incompletes_act6['run_id'] = run_id


        prep_incomplete_prod6_forecast_all.rename(columns={'forc_date':'Order_Create_Date'},inplace=True)
        incompletes_fcst.rename(columns={'cyear_month':'Order_Create_Date','fcst':'Order_Volume_(STD)'},inplace=True)

        df2=prep_incomplete_prod6_forecast_all[['Order_Create_Date', 'Order_Volume_(STD)', 'model', 'group_key', 'series_class','run_id','lag']]
        df3=incompletes_fcst[['Order_Create_Date', 'Order_Volume_(STD)', 'model', 'group_key', 'series_class','run_id','lag']]
        forecast_all_df = pd.concat([df1, df2,df3])
        all_actuals = pd.concat([prep_comp_prod.assign(series_class='ok'),incompletes_act,incompletes_act6])
        #### Serialize incompletes
        inc_dfs = {'incompletes_fcst':incompletes_fcst,'incompletes_act':incompletes_act}
        for dfname, df in inc_dfs.items():
            #print(dfname)
            df.to_pickle(f'results/runid{run_id}_{dfname}.pkl')
    else:
        forecast_all_df = df1
        all_actuals = prep_comp_prod.assign(series_class='ok')

    forecast_all_df[group_key] = forecast_all_df['group_key'].str.split('_',expand=True,n=len(group_key))

    #result= pd.concat([df1,df2,df3])

    #%% Final Transformations ###
    t_finalprep_start = time.time()
    #### Holdout accuracies and holdout info ######
    holdout_accs[group_key] = holdout_accs['group_key'].str.split('_',expand=True,n=len(group_key))
    #holdout_accs['product_no'] = np.where(holdout_accs['extra'].isnull(), holdout_accs['product_no'], holdout_accs['product_no']+'-'+ holdout_accs['extra'])
    holdout_accs['run_id'] = run_id
    #holdout_accs = pd.merge(holdout_accs.drop('extra',axis=1),prep_comp_prod[['product_no','product']].drop_duplicates(),how='inner',on='product_no')
    #holdout_accs = pd.merge(holdout_accs,prepped_df[['product_no','product']].drop_duplicates(),how='inner',on='product_no')

    outer_df_w = pd.merge(prods_mods_df,holdout_winners,how='left',on = ['model','group_key'])
    outer_df_w['winner'] = np.where(np.isnan(outer_df_w['smape']),False,True)
    #outer_df_w[['customer','jh_name_scm','ship_to_name','product_no']] = outer_df_w['group_key'].str.split('-',expand=True)
    outer_df_w[group_key] = outer_df_w['group_key'].str.split('_',expand=True,n=len(group_key))
    #outer_df_w['product_no'] = np.where(outer_df_w['extra'].isnull(), outer_df_w['product_no'], outer_df_w['product_no']+'-'+ outer_df_w['extra'])
    # Join with prep data to get product description
    #holdout_df = pd.merge(outer_df_w.drop('extra',axis=1),prep_comp_prod[['product_no','product']].drop_duplicates(),how='inner',on='product_no').rename(columns={'index':'cyear_month','fcst':'hold_fcst'})
    #holdout_df = pd.merge(outer_df_w,prepped_df[['product_no','product']].drop_duplicates(),how='inner',on='product_no').rename(columns={'index':'cyear_month','fcst':'hold_fcst'})
    holdout_df = outer_df_w.copy()
    holdout_df['run_id'] = run_id
    holdout_df['Order_Create_Date'] = holdout_df['Order_Create_Date'].dt.date
    holdout_df.rename(columns = {'Order_Volume_(STD)':'Holdout_Volume_(STD)'},inplace=True)
    holdout_info_df = pd.merge(holdout_df,prep_comp_prod[['Order_Create_Date','group_key','Order_Volume_(STD)']],on=['group_key','Order_Create_Date'])
    ## Set the product_no column to string so parquet can write
    #holdout_info_df['product_no'] = holdout_info_df['product_no'].astype(str)

    ##### Actuals and forecasts #####

    # Join with prep data to get product description
    forecast_final_df = forecast_all_df.copy()
    #forecast_final_df = pd.merge(forecast_all_df,prepped_df[['product_no','product']].drop_duplicates(),how='inner',on='product_no')
    # Concatenate actuals and forecasts #


    all_actuals['model'] = 'Actual'

    actuals_fcsts_df = pd.concat([forecast_final_df,all_actuals])
    #actuals_fcsts_df['product_no'] = actuals_fcsts_df['product_no'].astype(str)
    #actuals_fcsts_df['series_class'] = 'ok'


    #%% Final df ###

    actual_hold_df = pd.concat([all_actuals,holdout_info_df])

    final_df = pd.concat([actual_hold_df,forecast_final_df])
    #final_df['product_no'] = final_df['product_no'].astype(str)

    #gk = group_key.copy()
    #gk.extend(['product','cyear_month', 'group_key','model','run_id','lag','series_class'])
    #final_df_melt = final_df.melt(id_vars= gk,\
    #                              value_vars=['order_volume_kunits','hold_fcst','fcst'],var_name='series',value_name='volume_kunits')


    #%% Create a df with run details
    run_log = pd.DataFrame.from_dict({'run_id':[run_id],'run_date':[run_date],'run_scope':[run_scope],'run_type':[run_type],\
                                      'run_response':[run_response],'run_timegrain':[run_timegrain],'run_dimensions':[run_dimensions],\
                                          'run_datascientist':[run_datascientist],'run_asofdate':run_mofcst,
                                          'holdout_horizon':holdout_horizon, 'forecast_horizon':forecast_horizon})



    #%% Serialize important dataframes for research
    dfs = {'forecast_df':forecast_df, 'forecast_all_df':forecast_all_df, 'forecast_final_df':forecast_final_df, 'all_actuals':all_actuals,\
           'prepped_df':data, 'holdout_df':holdout_info_df,'holdout_winners':holdout_winners}
    for dfname, df in dfs.items():
        #print(dfname)
        df.to_pickle(f'results/runid{run_id}_{dfname}.pkl')

    t_finalprep_end = time.time()
    jhds_logger.info(f'End of final prep, it took {"{:.2f}".format((t_finalprep_end-t_finalprep_start)/60)} minutes')

    #%%% Write results to parquet , just in case ###
    ### Holdout_info_df and all_actuals are written to pkl above - probably need to standardize
    jhds_logger.info('Writing to parquet')
    df_list = [holdout_accs,actuals_fcsts_df,final_df,run_log]
    df_list_names = ['holdout_accs','actuals_forecasts_df','final_df','run_log']
    df_dict = dict(zip(df_list_names,df_list))

    w2d.write_res_parquet(run_id,df_dict)
    #%% Write results ###
    if write_file=='yes':
        t_writedb_start = time.time()
        jhds_logger.info('Writing to the db')
        #%%% Write results to db ###
        w2d.write_to_db(holdout_accs,name='holdout_accuracies',append=True)
        w2d.write_to_db(holdout_info_df,name='holdout_info',append=True)
        w2d.write_to_db(actuals_fcsts_df,name='actuals_forecasts',append=True)
        w2d.write_to_db(final_df,name='final',append=True)
        #w2d.write_to_db(ts_features_df, name= 'ts_features', append=True)
        w2d.write_to_db(run_log,name='run_log',append=True)
        w2d.write_to_db(all_actuals,name='all_actuals',append=True)

        t_writedb_end = time.time()
        jhds_logger.info(f'Finished writing to db, it took {"{:.2f}".format((t_writedb_end-t_writedb_start)/60)} minutes')

    t_script_end = time.time()
    jhds_logger.info(f'Finished script, it took {"{:.2f}".format((t_script_end-t_script_start)/60)} minutes')
    #%% End of main
    return None


#%% Call main

if __name__=="__main__":

    main()
