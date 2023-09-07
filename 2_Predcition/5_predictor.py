import dill
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os



def save_pkl(filepath, data):  # save data (model) as pkl format to filepath
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


data = pd.read_csv("./data/feature_data/STRUM_Q3421_fea_table_double.csv")  # 读取数据
data = data.drop("ID", axis=1)  # 删除ID列

rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
import shutil
shutil.copy('./resource/rfe_infos.xlsx', './resource/DDGWizard/rfe_infos.xlsx')

X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X
y = data["Experimental_DDG"].values  # 取出目标值 y

sc = StandardScaler()  # 定义标准化模型
X = sc.fit_transform(X)  # 标准化
save_pkl('./resource/DDGWizard/sc.pkl',sc)

BO_params= pd.read_excel("./resource/BO_Best_Param.xlsx")

model = XGBRegressor(n_estimators=int(BO_params['n_estimators']), max_depth=int(BO_params['max_depth']),
                     eta=float(BO_params['eta']), subsample=float(BO_params['subsample']),
                     colsample_bytree=float(BO_params['colsample_bytree']),
                     learning_rate=float(BO_params['learning_rate']), random_state=42)  # 定义XGBoost模型
model.fit(X, y)  # 训练模型

save_pkl("./resource/DDGWizard/predictor.pkl", model)


