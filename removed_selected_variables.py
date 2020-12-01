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
    # print()
    
    # fig = plt.figure()
    # plt.xlim(0,10)
    # plt.ylim(0,10)
    # plt.scatter(test['Violent_Crimes_Per_1000'], y_predict)
    # plt.xlabel("Crime Rate per 1000")
    # plt.ylabel("Predicted Crime Rate per 1000")
    # plt.title("Gradient Boost: %s" % title)
    
    return r2

#%% Load Data


def run_function(year_delay = 4):
    # year_delay = 4
    data = load_crime_education_data(year_delay)
    
    #%%
    
    data = data.drop(columns=['MURDER','RAPE','ROBBERY','AGGRAV','BURGLARY','LARCENY','MOTOR','ARSON','INDEX','%OF','STATE','IDCENSUS','FIPS','SCHLEV','WEIGHT', 'COUNTY', 'YRDAT', 'V33'])
    data = data.drop(columns=[
        'TOTALREV','TFEDREV','C14','C15','C16','C17','C19','B11','C20','C25','C36','B10','B12','B13','TSTREV','C01','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C24','C35','C38','C39','TLOCREV','T02','T06','T09','T15','T40','T99','D11','D23','A07','A08','A09','A11','A13','A15','A20','U97','TOTALEXP','TCURELSC','TCURINST','TCURSSVC','TCUROTH','NONELSEC','TCAPOUT','L12','M12','Q11','I86','Z32','Z34','_19H','_21F','_31F','_41F','_61V','_66V','W01','W31','W61'
    ])
    
    data["Violent_Crimes_Per_1000"] = data["Violent Crime"] / (data["POPULATION"] /1000)
    data = data.drop(columns = ['Violent Crime', "POPULATION"])
    
    data = data.fillna(0)
    
    #%% Test train split
    train, test = train_test_split(data,test_size=0.3,random_state=0)
    
    #%% Lasso 
    lasso_results = []
    # fit the results 
    start_it = 0
    iterations = 200
    
    for i in range(start_it ,iterations):    
        alpha = i/100000.0
        lasso_model = Lasso(alpha, fit_intercept = False, random_state = 0)
        lasso_model.fit(train.drop(columns=['Violent_Crimes_Per_1000'], axis=1), train['Violent_Crimes_Per_1000'])
        y_predict = lasso_model.predict(test.drop(columns=['Violent_Crimes_Per_1000'], axis=1))
        num_of_features = np.count_nonzero(lasso_model.coef_)
        # add
        lasso_results.append([num_of_features, r2_score(test['Violent_Crimes_Per_1000'], y_predict), alpha])
                              
    lasso_results = np.array(lasso_results)
    x = [i for i in range(start_it ,iterations)]
    
    best_alpha = lasso_results[lasso_results[:,1] == max(lasso_results[:,1])][0][2]
    
    #%%
    # best_alpha = 0.00114
    lasso_model = Lasso(best_alpha, fit_intercept = False, random_state = 0)
    lasso_model.fit(train.drop(columns=['Violent_Crimes_Per_1000'], axis=1), train['Violent_Crimes_Per_1000'])
    
    x = data.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
    x.columns[lasso_model.coef_ != 0]
    features_we_like = x.columns[lasso_model.coef_ != 0]
    
    betas = lasso_model.coef_[lasso_model.coef_ != 0]
    print([features_we_like[a] + ",%f"%betas[a] for a in range(0,len(features_we_like))])
    
    #%%    
    data_feature_selected = pd.concat([data[features_we_like], data['Violent_Crimes_Per_1000']], axis=1)

    #%% Dimensional reduction with standard PCA
    num_of_features = data_feature_selected.shape[1] - 1
    
    # results_of_pca = []
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

        # lr = LinearRegression()
        lr = GradientBoostingRegressor(
            learning_rate = .15,
            n_estimators = 1000,
            max_depth = 3,
            # criterion = "mse",
            random_state=0)
        lr.fit(X_train, y_train)
        y_predict = lr.predict(X_test)
        
        r2 = r2_score(y_test, y_predict)
        if r2 > best_r2_pca:
            best_r2_pca = r2
            best_pca_model = pca
            best_pca_data = data_pca
        
        # results_of_pca.append([i, r2])
    
    # results_of_pca = np.array(results_of_pca)
    
    #%%  
       
    num_of_features = data_feature_selected.shape[1] - 1   
    # results_of_nmf = []
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
        
        lr = GradientBoostingRegressor(
            learning_rate = .15,
            n_estimators = 1000,
            max_depth = 3,
            random_state=0)
        lr.fit(X_train, y_train)
        y_predict = lr.predict(X_test)
        r2 = r2_score(y_test, y_predict)
        
        if r2 > best_r2_nmf:
            best_r2_nmf = r2
            best_nmf_model = nmf
            best_nmf_data = data_nmf
        
        # results_of_nmf.append([i, r2])
    
    # results_of_nmf = np.array(results_of_nmf)
        
    #%% Gradient Boosting

    train, test = train_test_split(
        data_feature_selected,
        test_size=0.3,
        random_state=0)
    
    X_train = train.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
    y_train = train['Violent_Crimes_Per_1000']
    X_test = test.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
    y_test = test['Violent_Crimes_Per_1000']
    
    lasso_r2 = gradient_boosting(X_train, y_train, X_test, y_test, "Lasso")
    
    print("year Delay: %d" % year_delay)
    print("Gradient Boosted Results")
    print("Best Lasso R2: %f" % lasso_r2)
    print("Best PCA R2: %f" % best_r2_pca)
    print("Best NMF R2: %f" % best_r2_nmf)
    print("")
    
    results.append([year_delay, lasso_r2, best_r2_pca, best_r2_nmf])



results = []
# for year_delay in range(0,9):
#     run_function(year_delay)
    
run_function()    
print(results)


# #%% Intrepret SVM results
# data_columns = np.array(data_feature_selected.columns.array)
# for component in best_nmf_model.components_:
#     component = np.append(component, [0])
#     data_columns[component > 0]
#%%

results = np.array(results)

plt.subplot()
plt.plot(results[:,0],results[:,1], label ="Lasso")
plt.plot(results[:,0],results[:,2], label ="PCA")
plt.plot(results[:,0],results[:,3], label ="NMF")
plt.xlabel("Year Delay")
plt.ylabel("R Squared")
plt.legend()