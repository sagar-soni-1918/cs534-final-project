import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA, NMF

from sklearn.metrics import r2_score

from data_load import load_crime_education_data

#%% Function Definition

def gradient_boosting(X_train, y_train, X_test, y_test, title = ''):
    # this function takes in the train and test sets and a title to fit a gradient
    # boosted model
    
    gbr = GradientBoostingRegressor(
        learning_rate = .15,
        n_estimators = 1000,
        max_depth = 3,
        random_state=0)
    gbr.fit(X_train, y_train)
    y_predict = gbr.predict(X_test)
    r2 = r2_score(y_test, y_predict)
    
    return r2

#%% Load Data

def clean_data(data , variable_selection = None):
    # This function takes in the data and removes variables based on the variable selection setting
    # data is a pandas dataframe
    # variable selection can be 
        # none which is a standard removal of most revenue items, 
        # human_selected which removes all variables not selected by us, 
        # non_demo_variables which removes all varaibles which we believe can be influenced by the demographics
    # function returns the pandas dataframe with the variables removed and the crime rate normalized by crime per 1000 people

    data = data.drop(columns=['MURDER','RAPE','ROBBERY','AGGRAV','BURGLARY','LARCENY','MOTOR','ARSON','INDEX','%OF','STATE','IDCENSUS','FIPS','SCHLEV','WEIGHT', 'COUNTY', 'YRDAT', 'V33'])

    if variable_selection == None:
        data = data.drop(columns=[
            'TOTALREV','TFEDREV','TSTREV','TLOCREV','TOTALEXP','TCURELSC','TCURINST','TCURSSVC','TCUROTH','NONELSEC','TCAPOUT','Z32','Z34','_19H','_21F','_31F','_41F','_61V','_66V','W01','W31','W61'
        ])
    
    elif variable_selection == "human_selected":
        data = data.drop(columns=[
            '_19H','_21F','_31F','_41F','_61V','_66V','A07','A08','A09','A11','A13','A15','A20','B10','B11','B12','B13','C01','C04','C05','C07','C08','C10','C11','C12','C13','C14','C15','C20','C24','C25','C35','C36','C38','C39', 'D11','D23','F12','G15','I86','J07','J08','J09','J10','J11','J12','J13','J14','J17','J40','J45','J90','J96','J97','J98','J99','K09','K10','K11','L12','M12','NONELSEC','Q11','T02','T99','TCAPOUT','TCURELSC','TCURINST','TCUROTH','TCURSSVC','TFEDREV','TLOCREV','TOTALEXP','U22','U97','V10','V12','V14','V16','V18','V22','V24','V30','V32','V37','V38','V40','V45','V60','V65','V70','V75','V80','V85','V90','V91','V92','W01','W31','W61'
        ])

    elif variable_selection == "non_demo_variables":
        data = data.drop(columns=[
            'TOTALREV','TFEDREV','C14','C15','C16','C17','C19','B11','C20','C25','C36','B10','B12','B13','TSTREV','C01','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C24','C35','C38','C39','TLOCREV','T02','T06','T09','T15','T40','T99','D11','D23','A07','A08','A09','A11','A13','A15','A20','U97','TOTALEXP','TCURELSC','TCURINST','TCURSSVC','TCUROTH','NONELSEC','TCAPOUT','L12','M12','Q11','I86','Z32','Z34','_19H','_21F','_31F','_41F','_61V','_66V','W01','W31','W61'
        ])

    data["Violent_Crimes_Per_1000"] = data["Violent Crime"] / (data["POPULATION"] /1000)
    data = data.drop(columns = ['Violent Crime', "POPULATION"])
    
    data = data.fillna(0)

    return data

def run_lasso(data):
    # this function takes in the data pandas dataframe
    # runs a lasso regression on the dataset
    # returns a grandient boosted R_Squared 

    #%% Test train split
    train, test = train_test_split(data,test_size=0.3,random_state=0)
    
    #%% Lasso results stores the values for each iteration of lasso with a specific alpha
    # each index is number of features, R squared value, alpha for the model
    lasso_results = []
    # fit the results 
    start_it = 0
    iterations = 200
    
    for i in range(start_it ,iterations):    
        # we foound that the alpha for this dataset has to be very small
        alpha = i/100000.0
        lasso_model = Lasso(alpha, fit_intercept = False, random_state = 0)
        lasso_model.fit(train.drop(columns=['Violent_Crimes_Per_1000'], axis=1), train['Violent_Crimes_Per_1000'])
        y_predict = lasso_model.predict(test.drop(columns=['Violent_Crimes_Per_1000'], axis=1))
        num_of_features = np.count_nonzero(lasso_model.coef_)
        
        lasso_results.append([num_of_features, r2_score(test['Violent_Crimes_Per_1000'], y_predict), alpha])
                              
    lasso_results = np.array(lasso_results)
    x = [i for i in range(start_it ,iterations)]
    
    best_alpha = lasso_results[lasso_results[:,1] == max(lasso_results[:,1])][0][2]
    
    # fit a new model with the alpha with the higher R squared
    lasso_model = Lasso(best_alpha, fit_intercept = False, random_state = 0)
    lasso_model.fit(train.drop(columns=['Violent_Crimes_Per_1000'], axis=1), train['Violent_Crimes_Per_1000'])
    
    # find the titles of the features selected by the lasso model 
    x = data.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
    x.columns[lasso_model.coef_ != 0]
    features_we_like = x.columns[lasso_model.coef_ != 0]
    
    # print out a list that can be analyzed in excel
    betas = lasso_model.coef_[lasso_model.coef_ != 0]
    print([features_we_like[a] + ",%f"%betas[a] for a in range(0,len(features_we_like))])
    
    #%% create a new dataframe with the selected features 
    data_feature_selected = pd.concat([data[features_we_like], data['Violent_Crimes_Per_1000']], axis=1)

    train, test = train_test_split(
        data_feature_selected,
        test_size=0.3,
        random_state=0)
    
    # run a gradient boosted linear model
    X_train = train.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
    y_train = train['Violent_Crimes_Per_1000']
    X_test = test.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
    y_test = test['Violent_Crimes_Per_1000']
    
    lasso_r2 = gradient_boosting(X_train, y_train, X_test, y_test, "Lasso")

    return features_we_like, data_feature_selected, lasso_r2

def run_PCA(data_feature_selected):
    # Dimesnional reduction at times can show the interactions of variables
    # we wanted to look at the way that PCA would be able to provide a more accurate model 
    # which would also be easy to understand.
    # The function takes in the selected data features from the lasso model above
       
    # max reduction can only be the features - 1, otherwise no features are reduced
    num_of_features = data_feature_selected.shape[1] - 1
    
    # a 
    best_pca_model = None
    best_pca_data = None
    best_r2_pca = 0
    
    for i in range(1,num_of_features):
        
        pca = PCA(i)
        
        data_pca = pca.fit_transform(data_feature_selected.drop(columns=['Violent_Crimes_Per_1000'], axis=1))
        
        pca_beta = pca.components_
        
        X_train, X_test, y_train, y_test = train_test_split(
            data_pca,
            data_feature_selected['Violent_Crimes_Per_1000'],
            test_size=0.3,
            random_state=0)

        # fit a grandient boosted model to the reduced data
        lr = GradientBoostingRegressor(
            learning_rate = .15,
            n_estimators = 1000,
            max_depth = 3,
            # criterion = "mse",
            random_state=0)
        lr.fit(X_train, y_train)
        y_predict = lr.predict(X_test)
        
        # if the rsquared is better in this model, then select it otherwise disregrad 
        r2 = r2_score(y_test, y_predict)
        if r2 > best_r2_pca:
            best_r2_pca = r2
            best_pca_model = pca
            best_pca_data = data_pca
        
        # results_of_pca.append([i, r2])
    
    # results_of_pca = np.array(results_of_pca)
    return best_r2_pca, best_pca_model, best_pca_data

def run_NMF(data_feature_selected):
    # Dimesnional reduction at times can show the interactions of variables
    # we wanted to look at the way that PCA would be able to provide a more accurate model 
    # which would also be easy to understand.
    # The function takes in the selected data features from the lasso model above

    # max reduction can only be the features - 1, otherwise no features are reduced
    num_of_features = data_feature_selected.shape[1] - 1   
    
    best_nmf_model = None
    best_nmf_data = None
    best_r2_nmf = 0
    
    for i in range(1,num_of_features):
        
        nmf = NMF(i)
        
        data_nmf = nmf.fit_transform(data_feature_selected.drop(columns=['Violent_Crimes_Per_1000'], axis=1))
        
        nmf_beta = nmf.components_
        
        X_train, X_test, y_train, y_test = train_test_split(
            data_nmf,
            data_feature_selected['Violent_Crimes_Per_1000'],
            test_size=0.3,
            random_state=0)
        
        # fit a grandient boosted model to the reduced data
        lr = GradientBoostingRegressor(
            learning_rate = .15,
            n_estimators = 1000,
            max_depth = 3,
            random_state=0)
        lr.fit(X_train, y_train)
        y_predict = lr.predict(X_test)
        r2 = r2_score(y_test, y_predict)
        
        # if the rsquared is better in this model, then select it otherwise disregrad 
        if r2 > best_r2_nmf:
            best_r2_nmf = r2
            best_nmf_model = nmf
            best_nmf_data = data_nmf
        
    return best_r2_nmf, best_nmf_data, best_nmf_model


def run_function(year_delay = 4):
    # year_delay = 4
    data = load_crime_education_data(year_delay)
    
    data = clean_data(data, None)

    # Run a lasso model to do feature selection
    features_we_like, data_feature_selected, lasso_r2 = run_lasso(data)

    # Run PCA to see if feature reduction can yield better accuracy
    best_r2_pca, best_pca_model, best_pca_data = run_PCA(data_feature_selected)
       
    # Run NMF becasue functionally speaking, the model most have a positive or 0 relationship between all spending as all spending if postivie
    best_r2_nmf, best_nmf_data, best_nmf_model = run_NMF(data_feature_selected)
    
    # Print the R squared values
    print("year Delay: %d" % year_delay)
    print("Gradient Boosted Results")
    print("Best Lasso R2: %f" % lasso_r2)
    print("Best PCA R2: %f" % best_r2_pca)
    print("Best NMF R2: %f" % best_r2_nmf)
    print("")
    
    results.append([year_delay, lasso_r2, best_r2_pca, best_r2_nmf])

#%% Create a plot for the results when code is run with Time Delay

def plot_time_delay_results(results):
    plt.subplot()
    plt.plot(results[:,0],results[:,1], label ="Lasso")
    plt.plot(results[:,0],results[:,2], label ="PCA")
    plt.plot(results[:,0],results[:,3], label ="NMF")
    plt.xlabel("Year Delay")
    plt.ylabel("R Squared")
    plt.legend()

    return None

# if multi year delays wants to be run, then un comment the code below and comment out line 251 
results = []
# for year_delay in range(0,9):
#     run_function(year_delay)
# plot_time_delay_results(results)
    
run_function()    
print(results)



#%% Intrepret PCA results which can be taken to excel or alternative interpretation platforms
# data_columns = np.array(data_feature_selected.columns.array)
# for component in best_pca_model.components_:
#     component = np.append(component, [0])
#     data_columns[component > 0]

#%% Intrepret NMF results which can be taken to excel or alternative interpretation platforms
# data_columns = np.array(data_feature_selected.columns.array)
# for component in best_nmf_model.components_:
#     component = np.append(component, [0])
#     data_columns[component > 0]
