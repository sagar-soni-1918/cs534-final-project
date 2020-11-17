from utils import load_edu_spend_years, get_counties_from_dists

# %% load all data frames for desired years
years = range(1998,2002+1)
edu_spend_df_dict = load_edu_spend_years(years)

