
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




def importances_visualization(features_name, feature_importances, output_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=100)  # 定义画布
    ax.bar(features_name, feature_importances, width=0.5)  # 柱状图
    for x, y in zip(range(len(features_name)), [round(item, 4) for item in feature_importances]):
        ax.text(x=x, y=y, s=y, ha="center", va="bottom")  # 在柱状图上显示数字
    ax.set_title("Characteristic importance", fontsize=24)  # 标题
    ax.set_xlabel("Features name", fontsize=20)  # x轴标签
    ax.set_ylabel("Feature importances", fontsize=20)  # y轴标签
    ax.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    # ax.legend(loc="best", prop={"size": 16})  # 图例
    ax.set_xticks(range(len(features_name)), features_name, rotation=90)  # X轴刻度旋转90度
    # plt.tight_layout()  # 防重叠
    plt.savefig(output_path, bbox_inches="tight")  # 保存图像
    plt.show()  # 显示图像


data = pd.read_csv("./data/feature_data/STRUM_Q3421_fea_table_double.csv")  # 读取数据
data = data.drop("ID", axis=1)  # 删除ID列

rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X
y = data["Experimental_DDG"].values  # 取出目标值 y

sc = StandardScaler()  # 定义标准化模型
X = sc.fit_transform(X)  # 标准化


BO_params= pd.read_excel("./resource/BO_Best_Param.xlsx")



count=0
mse_list = []
rmse_list = []
mae_list = []
r2_list = []
pearson_list = []
spearman_list = []
feature_importances_recorder = []


for train_idxs, test_idxs in KFold(10, shuffle=True, random_state=42).split(X):  # 10折交叉验证
    count+=1
    X_train, X_test = X[train_idxs], X[test_idxs]  # 划分训练集和测试集
    y_train, y_test = y[train_idxs], y[test_idxs]  # 划分训练集和测试集
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
    feature_importances_recorder.append(model.feature_importances_)


evaluate_result = pd.DataFrame(
    np.array([mse_list, rmse_list, mae_list, r2_list, pearson_list, spearman_list]),
    index=["mse", "rmse", "mae", "r2", "pearson", "spearman"],
    columns=[f"split{i}_test_score" for i in range(10)],
)
evaluate_result["mean"] = evaluate_result.mean(axis=1)  # 计算平均评估结果
evaluate_result["std"] = evaluate_result.std(axis=1)  # 计算评估结果标准差
evaluate_result.to_excel("./evaluation/evaluate.xlsx", index=True)  # 保存评估结果

feature_importances = pd.DataFrame(
    np.array(feature_importances_recorder).T,
    index=X_cols,
    columns=[f"split{i}_test_score" for i in range(10)],
)  # 特征重要性
feature_importances["mean"] = feature_importances.mean(axis=1)  # 计算平均特征重要性
feature_importances.to_excel("./evaluation/feature_importances.xlsx", index=True)  # 保存特征重要性
importances_visualization(X_cols, feature_importances["mean"], "./evaluation/Feature_Importances.png")



print('average: ')
print("MSE:", sum(mse_list) / 10)
print("RMSE:", sum(rmse_list) / 10)
print("MAE:", sum(mae_list) / 10)
print("R^2 Score:", sum(r2_list) / 10)
print("pearson:", sum(pearson_list) / 10)
print("spearman:", sum(spearman_list) / 10)





# average
# 均方误差（MSE）: 1.5881857113908884
# 均方根误差（RMSE）: 1.256976295237982
# 平均绝对误差（MAE）: 0.8603480057380686
# 决定系数（R^2 Score）: 0.6927952586941212
# 皮尔逊相关系数（pearson）: 0.8352380015544089
# 斯皮尔曼相关系数（spearman）: 0.8274174551132678

