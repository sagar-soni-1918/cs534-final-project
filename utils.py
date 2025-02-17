import numpy as np
import pandas as pd
from difflib import get_close_matches

# python file containing all utility functions and dicts (loader, etc.)

# dictionary for district to county conversion
county_to_dict = {
    'Appling County School District':'Appling','Atkinson County School District':'Atkinson','Atlanta Public Schools':'DeKalb/Fulton','Bacon County School District':'Bacon','Baker County School District':'Baker','Baldwin County School District':'Baldwin','Banks County School District':'Banks','Barrow County Schools':'Barrow','Bartow County School District':'Bartow','Ben Hill County School District':'Ben Hill','Berrien County School District':'Berrien','Bibb County Public Schools':'Bibb','Bleckley County School District':'Bleckley','Brantley County School District':'Brantley','Bremen City School District':'Haralson','Brooks County School District':'Brooks','Bryan County School District':'Bryan','Buford City School District':'Gwinnett','Bulloch County School District':'Bulloch','Burke County School District':'Burke','Butts County School District':'Butts','Calhoun City School District':'Gordon','Calhoun County School District':'Calhoun','Camden County School District':'Camden','Candler County School District':'Candler','Carroll County School District':'Carroll','Carrollton City School District':'Carroll','Cartersville City School District':'Bartow','Catoosa County School District':'Catoosa','Charlton County School District':'Charlton','Chattahoochee County School District':'Chattahoochee','Chattooga County School District':'Chattooga','Cherokee County School District':'Cherokee','Chickamauga City School District':'Walker','Clarke County School District':'Clarke','Clay County School District':'Clay','Clayton County Public Schools':'Clayton','Clinch County School District':'Clinch','Cobb County Public Schools':'Cobb','Coffee County School District':'Coffee','Colquitt County School District':'Colquitt','Columbia County School System':'Columbia','Commerce City School District':'Jackson','Cook County School District':'Cook','Coweta County School System':'Coweta','Crawford County School District':'Crawford','Crisp County School District':'Crisp','Dade County School District':'Dade','Dalton City School District':'Whitfield','Dawson County School District':'Dawson','Decatur City School District':'DeKalb','Decatur County School District':'Decatur','DeKalb County School System':'DeKalb','Dodge County School District':'Dodge','Dooly County School District':'Dooly','Dougherty County School System':'Dougherty','Douglas County School District':'Douglas','Dublin City School District':'Laurens','Early County School District':'Early','Echols County School District':'Echols','Effingham County School District':'Effingham','Elbert County School District':'Elbert','Emanuel County School District':'Emanuel','Evans County School District':'Evans','Fannin County School District':'Fannin','Fayette County School System':'Fayette','Floyd County School District':'Floyd','Forsyth County Schools':'Forsyth','Franklin County School District':'Franklin','Fulton County School System':'Fulton','Gainesville City School District':'Hall','Gilmer County School District':'Gilmer','Glascock County School District':'Glascock','Glynn County School District':'Glynn','Gordon County School District':'Gordon','Grady County School District':'Grady','Greene County School District':'Greene','Griffin-Spalding County School District':'Spalding','Gwinnett County Public Schools':'Gwinnett','Habersham County School District':'Habersham','Hall County School District':'Hall','Hancock County School District':'Hancock','Haralson County School District':'Haralson','Harris County School District':'Harris','Hart County School District':'Hart','Heard County School District':'Heard','Henry County School District':'Henry','Houston County Schools':'Houston','Irwin County School District':'Irwin','Jackson County School District':'Jackson','Jasper County School District':'Jasper','Jeff Davis County School District':'Jeff Davis','Jefferson City School District':'Jackson','Jefferson County School District':'Jefferson','Jenkins County School District':'Jenkins','Johnson County School District':'Johnson','Jones County School District':'Jones','Lamar County School District':'Lamar','Lanier County School District':'Lanier','Laurens County School District':'Laurens','Lee County School District':'Lee','Liberty County School District':'Liberty','Lincoln County School District':'Lincoln','Long County School District':'Long','Lowndes County School District':'Lowndes','Lumpkin County School District':'Lumpkin','Macon County School District':'Macon','Madison County School District':'Madison','Marietta City School District':'Cobb','Marion County School District':'Marion','McDuffie County School District':'McDuffie','McIntosh County School District':'McIntosh','Meriwether County School District':'Meriwether','Miller County School District':'Miller','Mitchell County School District':'Mitchell','Monroe County School District':'Monroe','Montgomery County School District':'Montgomery','Morgan County School District':'Morgan','Murray County School District':'Murray','Muscogee County School District':'Muscogee','Newton County School District':'Newton','Oconee County School District':'Oconee','Oglethorpe County School District':'Oglethorpe','Paulding County School District':'Paulding','Peach County School District':'Peach','Pelham City School District':'Mitchell','Pickens County School District':'Pickens','Pierce County School District':'Pierce','Pike County School District':'Pike','Polk County School District':'Polk','Pulaski County School District':'Pulaski','Putnam County School District':'Putnam','Quitman County School District':'Quitman','Rabun County School District':'Rabun','Randolph County School District':'Randolph','Richmond County School System':'Richmond','Rockdale County School District':'Rockdale','Rome City School District':'Floyd','Savannah-Chatham County Public Schools':'Chatham','Schley County School District':'Schley','Screven County School District':'Screven','Seminole County School District':'Seminole','Social Circle City School District':'Walton','State schools':'Fulton','Stephens County School District':'Stephens','Stewart County School District':'Stewart','Sumter County School District':'Sumter','Talbot County School District':'Talbot','Taliaferro County School District':'Taliaferro','Tattnall County School District':'Tattnall','Taylor County School District':'Taylor','Telfair County School District':'Telfair','Terrell County School District':'Terrell','Thomas County School District':'Thomas','Thomaston-Upson County School District':'Upson','Thomasville City School District':'Thomas','Tift County School District':'Tift','Toombs County School District':'Toombs','Towns County School District':'Towns','Treutlen County School District':'Treutlen','Trion City School District':'Chattooga','Troup County School District':'Troup','Turner County School District':'Turner','Twiggs County School District':'Twiggs','Union County School District':'Union','Valdosta City School District':'Lowndes','Vidalia City School District':'Toombs','Walker County School District':'Walker','Walton County School District':'Walton','Ware County School District':'Ware','Warren County School District':'Warren','Washington County School District':'Washington','Wayne County School District':'Wayne','Webster County School District':'Webster','Wheeler County School District':'Wheeler','White County School District':'White','Whitfield County School District':'Whitfield','Wilcox County School District':'Wilcox','Wilkes County School District':'Wilkes','Wilkinson County School District':'Wilkinson','Worth County School District':'Worth'
}

first_crime_data_year = 2009
first_edu_data_year = 1998



# get all county matches for districts in the DataFrame
def get_counties_from_dists(edu_spend_df):
    lower_county_to_dict = dict((k.lower().split()[0], v) for k,v in county_to_dict.items())
    # get closest string match from the dictionary
    corresponding_county_list = []
    for iCountyIdx in edu_spend_df['NAME'].index:
        dist_key_match = get_close_matches(
            edu_spend_df['NAME'].loc[iCountyIdx].split()[0].lower(), # just get 1st word
            lower_county_to_dict.keys(),n=1,cutoff=0.2
            )
        corresponding_county = lower_county_to_dict[dist_key_match[0]]
        corresponding_county_list.append(corresponding_county)
    return corresponding_county_list

# load all DataFrames, select Georgia only, add 'Districts' column to each selected year
def load_edu_spend_years(years):
    all_data = {}
    edu_spend_df_dict = {}
    # input list or range of desired years, get dictionary of df's
    for iYear in years:
        all_data[iYear] = pd.read_csv(f'./education_data/elsec{str(iYear)[-2:]}.csv', low_memory=False)
        edu_spend_df_dict[iYear] = all_data[iYear][all_data[iYear]['STATE']==11] # get Georgia
        
        county_list = get_counties_from_dists(edu_spend_df_dict[iYear])
        
        edu_spend_df_dict[iYear].insert(loc=0,column='COUNTY',value=county_list)
        
        edu_spend_df_dict[iYear].astype(float, errors="ignore")
        
    return edu_spend_df_dict

# loads the dataframe for all of the Georgia Counties that we have data for
# Input is the last year we want to have data for
def load_crime_data(years):
    data = {}
    
    for iYear in range(first_crime_data_year,years+1):
        df = pd.read_csv('./Crime_Data/Georgia_Crime_data_%d.csv' % iYear, low_memory=False)

        df = df.astype(float , errors="ignore")
        data[iYear] = df
    
    return data


def education_by_county(education_dict): 
    for year in education_dict:    
        # education = edu_spend_df_dict['1998']    
        education = education_dict[year]
        df = education.copy(deep = True)
        a = df.groupby("COUNTY").sum()
        education_dict[year] = a
    
    return education_dict
    


def combine_crime_edu(crime, education, year_delta = 0):
    if year_delta == None:
        year_delta = 0
        
    crime_year = first_crime_data_year + year_delta
    education_year = first_edu_data_year
    
    data = crime[crime_year].join(education[education_year],on = "COUNTY", how="inner" )
    
    crime_year += 1
    education_year += 1
    
    # print(crime_year)    
    # print (education_year)
    while crime_year in crime and education_year in education:
        df = crime[crime_year].join(education[education_year],on = "COUNTY", how="inner" )
        data = data.append(df)
        
        # print(crime_year)    
        # print (education_year)
        
        crime_year = 1 + crime_year
        education_year = 1 + education_year

    return data
        
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    