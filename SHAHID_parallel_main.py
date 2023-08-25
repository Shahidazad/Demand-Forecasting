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
from tezcatli_scripts import load_data, utils, pre_process , write_to_database as w2d , parallel_functions as pf#, get_ts_features , fit_models,

### For tracking execution times
from os import path
import time
#import random
from random import getrandbits
import pickle

#For reading options from command line
import sys, getopt

import warnings
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [20,6]
pd.set_option('display.max_columns',12)
pd.set_option('display.width', 1000)

from joblib import Parallel, delayed, parallel_backend

from preprocessing_data import preprocessing_main02 


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
    RegressionModel,
    XGBModel)
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

write_file = 'No'

#%% Models to run ###
model_frames = ['Darts','Orbit']
dart_models = [ExponentialSmoothing(), NaiveSeasonal(), NaiveDrift(), NaiveMean(), AutoARIMA(), Theta(), FFT(), Prophet(),Croston(),LightGBMModel(lags=1),RandomForest(lags= 1,random_state=2309),RegressionEnsembleModel(forecasting_models=[ExponentialSmoothing(), NaiveSeasonal(),AutoARIMA(),TBATS()],regression_train_n_points=24),TBATS(),BATS(),RegressionModel(lags=1),RegressionModel(lags=10)]#, StandardRegressionModel()]
dart_models_names = ['ExponentialSmoothing', 'NaiveSeasonal','NaiveDrift','NaiveMean','AutoARIMA','Theta', 'FFT','Prophet','Croston','LightGBMModel','RandomForest','RegressionEnsembleModel','TBATS','BATS','RegressionModelL1','RegressionModelL10']

models = [DLT(), NaiveSeasonal(), NaiveDrift(), NaiveMean(), Prophet(),BATS(use_box_cox=False),RegressionModel(lags=1),XGBModel(lags=1)]
models_names = ['Orbit', 'NaiveSeasonal','NaiveDrift','NaiveMean','Prophet','BATS','RegressionModelL1','XGBoost']
orbit_models = [DLT()]
orbit_models_names = ['DampedLinearTrend']



# pankaj ji code
# models = [DLT(), ExponentialSmoothing(), NaiveSeasonal(), NaiveDrift(), NaiveMean(), AutoARIMA()]
models=[Prophet()]
# models_names = ['Orbit', 'ExponentialSmoothing', 'NaiveSeasonal','NaiveDrift' ,'NaiveMean','AutoARIMA']

models_names= ['Prophet']
# pankaj ji code
# models = [Prophet(),Croston(),LightGBMModel(lags=1),RandomForest(lags= 1,random_state=2309),TBATS(),BATS(),RegressionModel(lags=1)]

# models_names = ['Prophet','Croston','LightGBMModel','RandomForest','TBATS','BATS','RegressionModelL1'] 


print('------1.')
 #%% Main Method
def function_main1(forecast_group_name):
    
    keys, prep_comp_prod, prep_incomplete_prod = preprocessing_main02(forecast_group_name)
    #%%% Disable warnings
    warnings.filterwarnings("ignore")
    #%%% Set dates
    ## Set the date of the month to be run (one more than what data you have)
    run_mofcst = dt.datetime(run_config['current_year'],run_config['current_month'],1)
    train_date = run_mofcst + relativedelta(months=-holdout_horizon)
    order_min_date = dt.datetime(2015,4, 1).date()
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
  
    print('------1.1')
    if not test_run:
        orders = load_data.load_new_data()
        #w2d.write_new_data_db(orders)
        data = pd.concat([data,orders],ignore_index=True)
        data.to_parquet('data/tezcatli_orders_data.parquet')
    
    
    t_loaddata_end = time.time()
    jhds_logger.info(f'Finished loading data, it took {"{:.2f}".format((t_loaddata_end-t_loaddata_start)/60)} minutes')
    #### Finished loading data ###

    # This command will filter the data to only include rows where the region name is equal to North and the district name is equal to Mumbai. The filtered data will then be written to the file output.csv.
    #%%% Read parameters from CLI if any
    try:
        opts, args = getopt.getopt(sys.argv[1:],'r:d:s:mo:m:w:')
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    write_file = 'No'
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
    t_prepdata_end = time.time()
    jhds_logger.info(f'Finished prepping data, it took {"{:.2f}".format((t_prepdata_end-t_prepdata_start)/60)} minutes')

    #%% Fit best model ###
    print('------1.2.4')
    models_list = list(zip(models,models_names))
    keys_models = [(key,model) for key in keys for model in models_list]
    model_accs_dfs = []
    print('------1.2.5')
    
    timeNow_parallel0 = time.time()
    with parallel_backend('threading',n_jobs=2):
        print('------1.2.5.1')
        model_accs_dfs.append(Parallel()(delayed(pf.fit_models_parallel_keys)(item,prep_comp_prod, run_mofcst,train_date,jhds_logger) for item in keys_models if item is not None)) #,model_dfs, model_accs) for i in models_list))
    print('------1.2.6')
    timeNow_parallelEnd = time.time()
    print('Time taken by parallel code: ', timeNow_parallelEnd - timeNow_parallel0)

    '''    
    #%% Code in serial ===============================================
        model_accs_dfs2 = []
        print('------1.2.5')
        
        timeNow_serial0 = time.time()
        
        for item in keys_models:
            tempVar = pf.fit_models_parallel_keys(item, prep_comp_prod, run_mofcst,train_date, jhds_logger)
            model_accs_dfs2.append(tempVar)    
        
        timeNow_serialEnd = time.time()
        print('Time taken by parallel code: ', timeNow_serialEnd - timeNow_serial0)
    
    #%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ''' 
    prod_dfs, prod_accs = zip(*model_accs_dfs[0])
    accs_list_file = 'accs_list.pkl'
    preds_list_file = 'preds_list.pkl'
    with open(accs_list_file,'wb') as f:
        pickle.dump(prod_accs,f)
    with open(preds_list_file,'wb') as f:
        pickle.dump(prod_dfs,f)
    print('------1.2.7')
    jhds_logger.info('Pickled accuracies and predictions in holdout')
    prods_mods_df = pd.concat(prod_dfs).reset_index()
    prods_accs_df = pd.concat(prod_accs)

    ## Failed logging ##
    #failed_df = pd.DataFrame({'group_key':failed_keys})

    ###
    holdout_accs = prods_accs_df.set_index(['model','group_key']).reset_index()
    failed_accuracy = holdout_accs[holdout_accs['smape'].isnull()][['group_key','model']]
    failed_accuracy['fail_point'] = 'accuracy_calc'
    #failed_df = pd.concat([failed_df,failed_accuracy])
    holdout_accs = holdout_accs.dropna(subset=['smape'])
    
    #%% Get best model for every product ###
    holdout_winners = holdout_accs.loc[holdout_accs.groupby(['group_key'])['mase'].idxmin()]
    winners = dict(zip(holdout_winners['group_key'],holdout_winners['model']))

    ### Get holdout df

    #%% Generate forecasts for every key with best model ###
    ### Cycle through products ###
    forecasts = []
    t_fcst_start = time.time()
    jhds_logger.info('Start of forecasting')

    print('------1.3')
    model_fcsts_dfs = []
    with parallel_backend('threading',n_jobs=2 ):
        model_fcsts_dfs.append(Parallel()(delayed(pf.fcst_models_parallel_keys)(item,prep_comp_prod, run_mofcst,train_date,jhds_logger,forecast_horizon,winners) for item in keys_models if item is not None)) 
    
    print('------1.3.1')
    fcst_list_file = 'fcst_list.pkl'
    with open(fcst_list_file,'wb') as f:
        pickle.dump(forecasts,f)
    t_fcst_end = time.time()
    jhds_logger.info(f'End of Forecasting: run took {"{:.2f}".format((t_fcst_end-t_fcst_start)/60)} minutes')
    forecast_df = pd.concat(model_fcsts_dfs[0])
    forecast_df.reset_index(inplace=True)
    forecast_df['Order_Create_Date'] = forecast_df['Order_Create_Date'].dt.date
    ### Create a lag ###
    forecast_df['lag'] = forecast_df.groupby(['group_key','model'])['Order_Create_Date'].cumcount()
    ### Add run id ###
    forecast_df['run_id'] = run_id
    ### Add series class ###
    forecast_df['series_class'] = 'ok'

    #%% Post process forecasts , ensure quality ###
    #### Check for negative forecasts and turn them to 0
    forecast_df['Order_Volume_(STD)'] = np.where(forecast_df['Order_Volume_(STD)']<0,0,forecast_df['Order_Volume_(STD)'])
    df1=forecast_df[['Order_Create_Date', 'Order_Volume_(STD)', 'model', 'group_key', 'series_class','run_id','lag','winner']]
    print('after forecast df1 :=',df1)
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
            prep_incomplete_prod6_forecast_all['lag'] = prep_incomplete_prod6_forecast_all.groupby('group_key')['Order_Create_Date'].cumcount()#modify cyear_month to Order_Create_Date
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


        prep_incomplete_prod6_forecast_all.rename(columns={'forc_date':'Order_Create_Date','fcst':'Order_Volume_(STD)'},inplace=True)#rename fcst to Order_Volume(STD)
        incompletes_fcst.rename(columns={'Order_Create_Date':'Order_Create_Date','fcst':'Order_Volume_(STD)'},inplace=True)#modify cyear_month to Order_Create_Date

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
    print('------1.5')

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
    dfs = {'forecast_df':forecast_df, 
           'forecast_all_df':forecast_all_df, 
           'forecast_final_df':forecast_final_df, 
           'all_actuals':all_actuals,\
           'prepped_df':data, 
           'holdout_df':holdout_info_df,
           'holdout_winners':holdout_winners}
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
    print('------1.6')
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
    return forecast_final_df


X = function_main1('C+ St Plk')
