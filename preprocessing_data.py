import pandas as pd
from random import getrandbits
import datetime as dt
import pickle
from dateutil.relativedelta import relativedelta
from tezcatli_scripts import  utils, pre_process , write_to_database as w2d , parallel_functions as pf 


def preprocessing_main02(forecast_group_name):
    run_config = utils.read_params_in_from_json('run_config.json')
    run_mofcst = dt.datetime(run_config['current_year'],run_config['current_month'],1)
    order_min_date = dt.datetime(2015,4, 1).date()
    holdout_horizon = run_config['train_horizon']
    order_max_date = run_mofcst + relativedelta(months=-1)
    train_date = run_mofcst + relativedelta(months=-holdout_horizon)
    group_key = run_config['dimensions'].split('-')
    data=pd.read_parquet('data/tezcatli_orders_data.parquet')
    
    
    data['Order Create Date']=pd.to_datetime(data['Order Create Date'])
    print('columns name',data.columns)
    data=data.loc[data['forecast_group']==forecast_group_name] # IMP
    data = data[data['Region Name']!='Export/Interco']
    data = data[data['Product Segment'].isin(['Exterior','Interior'])]
    print("data after Product Segment'].isin(['Exterior','Interior'",data)
    
    data.columns = data.columns.str.replace(' - ', '_')
    data.columns = data.columns.str.replace(' ', '_')
    
    ### Replace nan value
    data[group_key] = data[group_key].fillna('NA')
    #data.replace(np.nan,'NA', inplace=True)
    
    ### Create grouping key
    data['group_key'] = data[group_key].agg('_'.join,axis=1)
    
    ### Replace original 0s with NAs
    # print("data before this data.dropna(subset=['Order_Volume_(STD)'],inplace=True)",data['Order_Create_Date'][-5:])
    data.dropna(subset=['Order_Volume_(STD)'],inplace=True)
    # print("data after this data.dropna(subset=['Order_Volume_(STD)'],inplace=True)",data['Order_Create_Date'][-5:])
    #data = data[data["Order_Volume_(STD)"]!=0]
    #data['Order_Volume_(STD)'] = np.where(data['Order_Volume_(STD)']==0,np.NAN,data['Order_Volume_(STD)'])
    
    
    print('------1.2')
    ###Use Fiscal year
    data['Fiscal_Year'] = data['Fiscal_YY'].replace('FY', '20', regex=True)
    print("data after this data['Fiscal_YY'].replace('FY', '20', regex=True)",data['Order_Create_Date'][-5:])
    data['order_date_fiscal'] = pd.to_datetime(data.Fiscal_Year + '/' + data.Fiscal_Period.astype(int).astype(str) + '/01')
    print("data after this ,data['Fiscal_YY'].replace('FY', '20', regex=True))",data)
    data = data.groupby(['order_date_fiscal','group_key'])['Order_Volume_(STD)'].sum().reset_index()
    data.rename(columns = {'order_date_fiscal':'Order_Create_Date'}, inplace = True)
    
    
    
    ### Group by month and group key
    #data = data.set_index('Order_Create_Date').groupby([pd.Grouper(freq='MS'),'group_key'])['Order_Volume_(STD)'].sum().reset_index()
    #data = data.groupby(['Order_Create_Date','group_key'])['Order_Volume_(STD)'].sum().reset_index()
    
    ### Filter for orders until month of forecast
    data = data[data['Order_Create_Date']<=run_mofcst]
    run_id = getrandbits(32)
    ### Filter for orders greater than min date
    data = data[data['Order_Create_Date'].dt.date>=order_min_date]
    # print('columns name',data.columns)
    # data=data.loc[data['forecast_group']=="C+ St Plk"] # IMP
    # print('data before forecast ',data)
    # print("Length of data :",len(data))
    
    ### Create keys for date indexing
    allkeys = data['group_key'].unique()
    
    ### Fix the dates
    idx = pd.date_range(order_min_date,order_max_date,freq='MS',name='Order_Create_Date')
    # print('data before preprocessing ',data)
    data = pre_process.fix_dates(data,idx,allkeys)
    print('order_min_date :-',order_min_date)
    print('order_max_date:-',order_max_date)
    print('train_date : -',train_date)
    print('run_mofcst:-',run_mofcst)
    ### Get counts of values higher than 0 ###
    data_counts = data[data['Order_Volume_(STD)']>0.000001][['group_key']].value_counts().reset_index(name='counts')
    ### Find complete and incomplete data
    print('------1.2.1')
    incomplete_data_df = data_counts.query('counts<=6')
    #incomplete_data_df = data.iloc[incomplete_data_idx]
    print('------1.2.2')
    complete_data_df = data_counts.query('counts>6')
    # print('data before merge',data)
    prep_init_prod = data.merge(complete_data_df[['group_key']].drop_duplicates(),how='inner',on='group_key')
    prep_init_prod['run_id'] = run_id
    
    prep_incomplete_prod = data.merge(incomplete_data_df[['group_key']].drop_duplicates(),how='inner',on='group_key')
    prep_incomplete_prod['run_id'] = run_id
    prep_incomplete_prod['fail_point'] = 'insufficient data'
    
    print('------1.2.3')
    
    keys = prep_init_prod['group_key'].unique()
    
    ### Group by date and group key
    #prep_init_prod = prep_init_prod.groupby(['Order_Create_Date','group_key'])['Order_Volume_(SQFT)'].sum().reset_index()
    
    #%% Pickle the prepped data
    prep_init_file = 'prep_init.pkl'
    with open(prep_init_file,'wb') as f:
        pickle.dump(prep_init_prod,f)
    
    #%% Dataframe for modeling
    prep_comp_prod = prep_init_prod[prep_init_prod['group_key'].isin(keys)]
    
    #prep_comp_prod = prep_init_prod.copy()
    
    return keys, prep_comp_prod, prep_incomplete_prod
