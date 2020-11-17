import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score


# from sklearn.datasets import load_breast_cancer
from data_load import load_crime_education_data

#%% Load Data
data = load_crime_education_data(5)

#%%

data = data.drop(columns=['MURDER','RAPE','ROBBERY','AGGRAV','BURGLARY','LARCENY','MOTOR','ARSON','INDEX','%OF','STATE','IDCENSUS','FIPS','SCHLEV','WEIGHT', 'COUNTY', 'YRDAT', 'V33'])
data = data.drop(columns=[
    'TOTALREV','TFEDREV','TSTREV','TLOCREV','TOTALEXP','TCURELSC','TCURINST','TCURSSVC','TCUROTH','NONELSEC','TCAPOUT','Z32','Z34','_19H','_21F','_31F','_41F','_61V','_66V','W01','W31','W61'
])


# data = data.astype(float, errors = "ignore")

data["Violent_Crimes_Per_1000"] = data["Violent Crime"] / (data["POPULATION"] /1000)
data = data.drop(columns = ['Violent Crime', "POPULATION"])

data = data.fillna(0)

#%% Test train split
train, test = train_test_split(
    data,
    test_size=0.3,
    random_state=0)



#%% Lasso 
lasso_results = []
# fit the results 
start_it = 0
iterations = 500

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

fig, axs = plt.subplots(1,2)
ax1, ax2 = axs.flatten()

# ax1.plot(x,lasso_results[:,0])
# ax1.set_ylabel("Number of features")
# ax1.set_xlabel("Alpha")

ax1.plot(lasso_results[:,2],lasso_results[:,1])
ax1.set_ylabel("R2")
ax1.set_xlabel("Alpha")

ax2.plot(lasso_results[:,0], lasso_results[:,1])
# ax2.set_ylabel("R2")
ax2.set_xlabel("# Features")

best_alpha = lasso_results[lasso_results[:,1] == max(lasso_results[:,1])][0][2]

#%%
# best_alpha = 0.00114
lasso_model = Lasso(best_alpha, fit_intercept = False, random_state = 0)
lasso_model.fit(train.drop(columns=['Violent_Crimes_Per_1000'], axis=1), train['Violent_Crimes_Per_1000'])

x = data.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
x.columns[lasso_model.coef_ != 0]
features_we_like = x.columns[lasso_model.coef_ != 0]

fig = plt.figure(figsize=(2^16,2^16))
plt.bar(features_we_like, lasso_model.coef_[lasso_model.coef_ != 0])
fig = plt.figure()
plt.xlim(0,10)
plt.ylim(0,10)
plt.scatter(test['Violent_Crimes_Per_1000'], y_predict)

betas = lasso_model.coef_[lasso_model.coef_ != 0]
print([features_we_like[a] + ",%f"%betas[a] for a in range(0,len(features_we_like))])
plt.xlabel("Crime Rate per 1000")
plt.ylabel("Predicted Crime Rate per 1000")
plt.title("Lasso")

#%%

data_feature_selected = pd.concat([data[features_we_like], data['Violent_Crimes_Per_1000']], axis=1)


#%% Dimensional reduction with standard PCA

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

num_of_features = data_feature_selected.shape[1] - 1

results_of_pca = []

for i in range(1,num_of_features):
    
    pca = PCA(i)
    
    data_pca = pca.fit_transform(data_feature_selected.drop(columns=['Violent_Crimes_Per_1000'], axis=1))
    
    pca_beta = pca.components_
    
    train, test = train_test_split(
        data_feature_selected,
        test_size=0.3,
        random_state=0)
    
    X_train = train.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
    y_train = train['Violent_Crimes_Per_1000']
    X_test = test.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
    y_test = test['Violent_Crimes_Per_1000']
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    
    results_of_pca.append([i, r2_score(y_test, y_predict)])

results_of_pca = np.array(results_of_pca)

fig = plt.figure()
plt.plot(results_of_pca[:,0], results_of_pca[:,1])
plt.title("PCA")
plt.xlabel("Feature Selection")
plt.ylabel("R2")
    
    
    
#%%  
    
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression

num_of_features = data_feature_selected.shape[1] - 1

results_of_nmf = []

for i in range(1,num_of_features):
    
    nmf = NMF(i)
    
    data_nmf = nmf.fit_transform(data_feature_selected.drop(columns=['Violent_Crimes_Per_1000'], axis=1))
    
    nmf_beta = nmf.components_
    
    train, test = train_test_split(
        data_feature_selected,
        test_size=0.3,
        random_state=0)
    
    X_train = train.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
    y_train = train['Violent_Crimes_Per_1000']
    X_test = test.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
    y_test = test['Violent_Crimes_Per_1000']
    # X_train = np.nan_to_num(X_train)
    # X_test = np.nan_to_num(X_test)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    
    results_of_nmf.append([i, r2_score(y_test, y_predict)])

results_of_nmf = np.array(results_of_nmf)

fig = plt.figure()
plt.plot(results_of_nmf[:,0], results_of_nmf[:,1])   
plt.title("NMF")
plt.xlabel("Feature Selection")
plt.ylabel("R2")
    
#%% Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor



train, test = train_test_split(
    data_feature_selected,
    # data.drop(columns=['Violent_Crimes_Per_1000'], axis=1),
    # data['Violent_Crimes_Per_1000'],
    # data_pca,
    # data.target,
    test_size=0.3,
    random_state=0)

X_train = train.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
y_train = train['Violent_Crimes_Per_1000']
X_test = test.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
y_test = test['Violent_Crimes_Per_1000']

gbr = GradientBoostingRegressor(
    learning_rate = .15,
    n_estimators = 1000,
    max_depth = 3,
    # criterion = "mse",
    random_state=0)
gbr.fit(X_train, y_train)
y_predict = gbr.predict(X_test)
print(r2_score(y_test, y_predict))

fig = plt.figure()
plt.xlim(0,10)
plt.ylim(0,10)
plt.scatter(test['Violent_Crimes_Per_1000'], y_predict)
plt.xlabel("Crime Rate per 1000")
plt.ylabel("Predicted Crime Rate per 1000")
plt.title("Gradient Boost")
    
#%% Bagggings
# from sklearn.ensemble import BaggingRegressor


# train, test = train_test_split(
#     data_feature_selected,
#     # data.drop(columns=['Violent_Crimes_Per_1000'], axis=1),
#     # data['Violent_Crimes_Per_1000'],
#     # data_pca,
#     # data.target,
#     test_size=0.3,
#     random_state=0)

# X_train = train.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
# y_train = train['Violent_Crimes_Per_1000']
# X_test = test.drop(columns=['Violent_Crimes_Per_1000'], axis=1)
# y_test = test['Violent_Crimes_Per_1000']



# br = BaggingRegressor(
#     # max_samples = 10,
#     max_features = X_train.shape[1] - 5,
#     random_state = 0 
#     )
    
# br.fit(X_train, y_train)
# y_predict = br.predict(X_test)
# print(r2_score(y_test, y_predict))

# fig = plt.figure()
# plt.xlim(0,10)
# plt.ylim(0,10)
# plt.scatter(test['Violent_Crimes_Per_1000'], y_predict)
    
    
    
    
    
    
    
    
