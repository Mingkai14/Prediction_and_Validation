import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

#load data and drop ID column
df = pd.read_csv('./data/feature_data/STRUM_Q3421_fea_table_double.csv')
df.drop('ID', axis=1, inplace=True)


#obtain target value
Y_reg = list(df['Experimental_DDG'])
Y_cls = list(df['Experimental_DDG_Classification'])

#remove target column
df.drop(['Experimental_DDG', 'Experimental_DDG_Classification'], axis=1, inplace=True)
fea = df.values

# standardize features value
sc = StandardScaler()
fea = sc.fit_transform(fea)

X = np.array(fea)
y = np.array(Y_reg)


kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# train models in different dataset
count=0
mse_list = []
rmse_list = []
mae_list = []
r2_list = []
pearson_list = []
spearman_list = []

for train_idx, test_idx in kfold.split(X):
    count+=1
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # create model and train
    model = XGBRegressor(random_state=42)

    model.fit(X_train, y_train)

    #predict to obtain prediction value
    y_pred = model.predict(X_test)

    #evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    y_test = np.array(y_test).reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))
    yy = np.concatenate([y_test, y_pred], -1)
    yy = yy.T
    corr_matrix = np.corrcoef(yy)
    pearson=corr_matrix[0][1]

    correlation, p_value = spearmanr(y_test, y_pred)
    spearman=correlation


    # print results
    print(f'cross {count}:')
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R^2 Score:", r2)
    print("pearson:", pearson)
    print("spearman:", spearman)

    mse_list.append(mse)
    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)
    pearson_list.append(pearson)
    spearman_list.append(spearman)

print('average: ')
print("MSE:", sum(mse_list)/10)
print("RMSE:", sum(rmse_list)/10)
print("MAE:", sum(mae_list)/10)
print("R^2 Score:", sum(r2_list)/10)
print("Pearson:", sum(pearson_list)/10)
print("Spearman:", sum(spearman_list)/10)

# average
# MSE: 1.8204568342826355
# RMSE: 1.3446923810122637
# MAE: 0.9278109863535311
# R^2 Score: 0.6469635266416581
# pearson: 0.8044803944281815
# spearman: 0.7935926384870521
