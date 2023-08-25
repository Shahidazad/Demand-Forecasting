

import pandas as pd 
from SHAHID_parallel_main import function_main1


forecast_groups  = ["C+ St Plk",
                    "C+ St NT3",
                    "Pr Plk",
                    "Pr HLD",
                    "Pr Pnl",
                    "C+ St Pnl",
                    "C+ St HTG",
                    "Pr Soff 12'",
                    "Pr CemPre",
                    "Pr NT3",
                    "C+ St HLD",
                    "Pr Soff 8'"]

forecast_groups  = ["Pr Soff 8'"]
final_df_list = []

for i in range(len(forecast_groups)):
    temp_df = function_main1(forecast_groups[i])
    final_df_list.append(temp_df)

result = pd.concat(final_df_list)

result.to_csv('PANKAJ_ALL_CONCATNATED.csv')