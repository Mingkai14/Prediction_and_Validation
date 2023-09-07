import dill
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def load_pkl(filepath):  # load model of pkl format from filepath, and return data (model)
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data

data = pd.read_csv("./data/feature_data/Ssym_fea_table_forward.csv")  # 读取数据
data = data.drop("ID", axis=1)  # 删除ID列

rfe_infos = pd.read_excel("./resource/DDGWizard/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X
y_test = data["Experimental_DDG"].values  # 取出目标值 y

sc=load_pkl('./resource/DDGWizard/sc.pkl')
X = sc.transform(X)  # 标准化

model=load_pkl('./resource/DDGWizard/predictor.pkl')
y_pred = model.predict(X)

pred_ddg_both = pd.DataFrame(
    y_pred,
    columns=['pred_ddg'],
)

pred_ddg_both.to_excel("./evaluation_Ssym/pred_ddg_forward.xlsx", index=True)


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

print(mse)
print(rmse)
print(mae)
print(r2)
print(pearson)
print(spearman)

evaluate_result = pd.DataFrame(
    np.array([mse,rmse,mae,r2,pearson,spearman]),
    index=["mse", "rmse", "mae", "r2", "pearson", "spearman"],
    columns=['prediction_on_Ssym_forward'],
)

evaluate_result.to_excel("./evaluation_Ssym/evaluate_forward.xlsx", index=True)  # 保存评估结果



