# %%
import numpy as np
import pandas as pd

# %% function for loading in desired years of edu_spend data
def load_edu_spend_years(years):
    edu_spend_df_dict = {}
    # input list or range of desired years, get dictionary of df's
    for iYear in years:
        edu_spend_df_dict[str(iYear)] = pd.read_csv(f'./education_data/elsec{str(iYear)[-2:]}.csv')
    return edu_spend_df_dict
    
# %% load all data frames for desired years
years = range(1998,2002+1)
edu_spend_df_dict = load_edu_spend_years(years)
