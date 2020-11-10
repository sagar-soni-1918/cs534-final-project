import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()



X_train, X_test, y_train, y_test = train_test_split(
    # data.drop(labels=['target', 'ID'], axis=1),
    # data['target'],
    data.data,
    data.target,
    test_size=0.3,
    random_state=0)

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)



#%% Lasso 
# lasso_results = []
# # fit the results 
# iterations = 50
# for i in range(0,iterations):    
#     lasso_model = Lasso(alpha = i/10.0, fit_intercept = False, random_state = 0)
#     lasso_model.fit(X_train, y_train)
#     y_predict = lasso_model.predict(X_test)
#     num_of_features = np.count_nonzero(lasso_model.coef_)
#     # add
#     lasso_results.append([num_of_features, r2_score(y_test, y_predict)])
                          
# lasso_results = np.array(lasso_results)
# x = [i/10.0 for i in range(0,iterations)]

# fig, axs = plt.subplots(2,2)
# ax1, ax2, ax3, ax4 = axs.flatten()

# ax1.plot(x,lasso_results[:,0])
# ax1.set_ylabel("Number of features")
# ax1.set_xlabel("Alpha")

# ax2.plot(x,lasso_results[:,1])
# ax2.set_ylabel("R2")
# ax2.set_xlabel("Alpha")

# ax3.plot(lasso_results[:,0], lasso_results[:,1])
# ax3.set_ylabel("R2")
# ax3.set_xlabel("# Features")


#%% Dimensional reduction with standard PCA

# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression

# num_of_features = data.data.shape[1]

# results_of_pca = []

# for i in range(1,num_of_features):
    
#     pca = PCA(i)
    
#     data_pca = pca.fit_transform(data.data)
    
#     pca_beta = pca.components_
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         # data.drop(labels=['target', 'ID'], axis=1),
#         # data['target'],
#         data_pca,
#         data.target,
#         test_size=0.3,
#         random_state=0)
    
#     X_train = np.nan_to_num(X_train)
#     X_test = np.nan_to_num(X_test)
    
#     lr = LinearRegression()
#     lr.fit(X_train, y_train)
#     y_predict = lr.predict(X_test)
    
#     results_of_pca.append([i, r2_score(y_test, y_predict)])

# results_of_pca = np.array(results_of_pca)

# plt.plot(results_of_pca[:,0], results_of_pca[:,1])
    
    
    
#%%  
    
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression

num_of_features = data.data.shape[1]

results_of_nmf = []

for i in range(1,num_of_features):
    
    nmf = NMF(i)
    
    data_nmf = nmf.fit_transform(data.data)
    
    nmf_beta = nmf.components_
    
    X_train, X_test, y_train, y_test = train_test_split(
        # data.drop(labels=['target', 'ID'], axis=1),
        # data['target'],
        data_nmf,
        data.target,
        test_size=0.3,
        random_state=0)
    
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    
    results_of_nmf.append([i, r2_score(y_test, y_predict)])

results_of_nmf = np.array(results_of_nmf)

plt.plot(results_of_nmf[:,0], results_of_nmf[:,1])   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
