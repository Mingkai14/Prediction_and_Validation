
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score






data_train = pd.read_csv("./data/feature_data/S2298_fea_table_single.csv")  # 读取数据
data_test = pd.read_csv("./data/feature_data/S350_fea_table_single.csv")
data_train = data_train.drop("ID", axis=1)  # 删除ID列
data_test = data_test.drop("ID", axis=1)  # 删除ID列

rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X_train = data_train[X_cols].values  # 取出特征值 X
X_test = data_test[X_cols].values  # 取出特征值 X
y_train = data_train["Experimental_DDG"].values  # 取出目标值 y
y_test = data_test["Experimental_DDG"].values  # 取出目标值 y



sc = StandardScaler()  # 定义标准化模型
X_train = sc.fit_transform(X_train)  # 标准化
X_test=sc.transform(X_test)




BO_params= pd.read_excel("./resource/BO_Best_Param.xlsx")



model = XGBRegressor(n_estimators=int(BO_params['n_estimators']),max_depth=int(BO_params['max_depth']),eta=float(BO_params['eta']),subsample=float(BO_params['subsample']),colsample_bytree=float(BO_params['colsample_bytree']),learning_rate=float(BO_params['learning_rate']),random_state=42)  # 定义XGBoost模型
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


print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R^2 Score:", r2)
print("pearson:", pearson)
print("spearman:", spearman)

print("RMSE:", rmse)
print("pearson:", pearson)



# average
# 均方根误差（RMSE）: 0.9729975651627084
# 皮尔逊相关系数（pearson）: 0.7951493281068381


