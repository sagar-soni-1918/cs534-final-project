import utils as ut
import pandas as pd


# %% load all data frames for desired years

def load_crime_education_data(year_delta = 0):
    years = range(1998,2002+1)
    edu_spend_df_dict = ut.load_edu_spend_years(years)
    crime_dict = ut.load_crime_data(2017)
    
    # convert all int to float
    # edu_spend_df_dict = pd.to_numeric(edu_spend_df_dict)
    # crime_dict = pd.to_numeric(crime_dict)
    
    education = ut.education_by_county(edu_spend_df_dict)
    education = scale_education_by_rev(education)
    
    data = ut.combine_crime_edu(crime_dict, education, year_delta)
    
    return data


def scale_education_by_rev(education_dict):
    for iYear in education_dict: 
        education = education_dict[iYear]
        education = education.astype(float, errors="ignore")
        education = education.div(education['TOTALREV'],0)
        education 
        
        education_dict[iYear] = education
        
    
        
    return education_dict