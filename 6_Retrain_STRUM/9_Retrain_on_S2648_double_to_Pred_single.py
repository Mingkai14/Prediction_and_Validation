
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import copy

def exclude(data_excluded:list,data_input:list):
    exclude_ids=[]
    for row in data_excluded:
        id=row[0]
        exclude_ids.append(id)


    for i in range(len(data_input)-1,-1,-1):
        if data_input[i][0] in exclude_ids:
            data_input.pop(i)



rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合


BO_params= pd.read_excel("./resource/BO_Best_Param.xlsx")

data_single = pd.read_csv("./data/feature_data/S2648_fea_table_single.csv")  # 读取数据
data_double = pd.read_csv("./data/feature_data/S2648_fea_table_double.csv")  # 读取数据


X = data_single[X_cols].values.tolist()  # 取出特征值 X

sc = StandardScaler()  # 定义标准化模型
sc.fit(X)  # 标准化

X_cols=['ID']+X_cols+["Experimental_DDG"]

data_single_list=data_single[X_cols].values.tolist()
data_double_list=data_double[X_cols].values.tolist()






count=0
rmse_list = []
pearson_list = []



for train_idxs, test_idxs in KFold(5, shuffle=True, random_state=42).split(data_single_list):  # 10折交叉验证
    count+=1
    test_data_list=[]
    for test_idx in test_idxs:
        test_data_list.append(data_single_list[test_idx])
    temp_list=copy.deepcopy(data_double_list)
    exclude(test_data_list,temp_list)
    train_data_list=temp_list


    for data in test_data_list:
        assert isinstance(data,list)
        data.pop(0)

    for data in train_data_list:
        assert isinstance(data,list)
        data.pop(0)

    y_train=[]
    y_test=[]

    for data in train_data_list:
        assert isinstance(data,list)
        ddg=data.pop(-1)
        y_train.append(ddg)

    for data in test_data_list:
        assert isinstance(data,list)
        ddg=data.pop(-1)
        y_test.append(ddg)

    X_train=train_data_list
    X_test=test_data_list
    sc.transform(X_train)
    sc.transform(X_test)

    model = XGBRegressor(n_estimators=int(BO_params['n_estimators']), max_depth=int(BO_params['max_depth']),
                         eta=float(BO_params['eta']), subsample=float(BO_params['subsample']),
                         colsample_bytree=float(BO_params['colsample_bytree']),
                         learning_rate=float(BO_params['learning_rate']), random_state=42)  # 定义XGBoost模型
    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 预测

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    y_test = np.array(y_test).reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))
    yy = np.concatenate([y_test, y_pred], -1)
    yy = yy.T
    corr_matrix = np.corrcoef(yy)
    pearson = corr_matrix[0][1]

    correlation, p_value = spearmanr(y_test, y_pred)
    spearman = correlation

    print(f'cross {count}:')
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R^2 Score:", r2)
    print("pearson:", pearson)
    print("spearman:", spearman)

    rmse_list.append(rmse)
    pearson_list.append(pearson)

print('average: ')
print("RMSE:", sum(rmse_list) / 5)
print("pearson:", sum(pearson_list) / 5)



# RMSE: 0.9401104783286491
# pearson: 0.7699984747590023









