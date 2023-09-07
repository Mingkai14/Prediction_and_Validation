import dill
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def save_pkl(filepath, data):  # save data (model) as pkl format to filepath
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):  # load model of pkl format from filepath, and return data (model)
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


if __name__ == "__main__":
    data = pd.read_csv("./data/feature_data/STRUM_Q3421_fea_table_double.csv")  # read fearures_table
    data = data.drop("ID", axis=1)  # drop ID column
    X = data.drop(columns=["Experimental_DDG", "Experimental_DDG_Classification"], axis=1).copy()  # drop target value
    y = data["Experimental_DDG"].tolist()  # obtain target value

    sc = StandardScaler()  # Standardize
    X[X.columns] = sc.fit_transform(X[X.columns].values)

    model = XGBRegressor(random_state=42)  # define XGBoost model
    rfecv = RFECV(                         # define RFE model
        estimator=model,
        step=1,
        min_features_to_select=1,  # set minimum of features as 1
        cv=KFold(10, shuffle=True, random_state=42),  # 10-fold cross-validation
        scoring="r2",  # use R-squared as score metrics
        verbose=100,
        n_jobs=-1,
    )
    rfecv.fit(X, y)  # train rfe to select features
    save_pkl("./resource/rfecv.pkl", rfecv)  # save rfe model


    rfecv = load_pkl("./resource/rfecv.pkl")  # load rfe model

    print("features num: ", rfecv.n_features_in_)  # 总特征数量
    print("best features num: ", rfecv.n_features_)  # 最佳特征数量

    rfe_infos = pd.DataFrame(
        {
            "feature_names": rfecv.feature_names_in_,
            "ranking": rfecv.ranking_,
            "support": rfecv.support_,
        }
    )  # RFE模型的特征信息
    rfe_infos.to_excel("./resource/rfe_infos.xlsx", index=False)  # 保存RFE模型的特征信息

    rfe_results = pd.DataFrame(rfecv.cv_results_)  # RFE模型的结果
    rfe_results.to_excel("./evaluation/rfe_results.xlsx", index=False)  # 保存RFE模型的结果

    # 画图
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=100)  # 定义画布
    ax.plot(
        range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
        rfecv.cv_results_["mean_test_score"],
        marker="",
        linestyle="-",
        linewidth=2,
        label=f"mean_test_score",
    )  # 各交叉验证均值结果曲线
    for cv_idx in range(len(rfecv.cv_results_) - 2):
        ax.plot(
            range(1, len(rfecv.cv_results_[f"split{cv_idx}_test_score"]) + 1),
            rfecv.cv_results_[f"split{cv_idx}_test_score"],
            marker="",
            linestyle="-",
            linewidth=2,
            label=f"split{cv_idx}_test_score",
        )  # 各交叉验证结果曲线
    ax.set_title("RFE feature screening curve", fontsize=24)  # 标题
    ax.set_xlabel("Features num", fontsize=20)  # x轴标签
    ax.set_ylabel("R2", fontsize=20)  # y轴标签
    ax.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    ax.legend(loc="best", prop={"size": 16})  # 图例
    plt.savefig("./evaluation/rfe_graph.png")  # 保存图像
    # plt.show()  # 显示图像


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=100)  # 定义画布
    ax.plot(
        range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
        rfecv.cv_results_["mean_test_score"],
        marker="",
        linestyle="-",
        linewidth=2,
        label=f"mean_test_score",
    )  # 各交叉验证均值结果曲线
    ax.set_title("RFE feature screening curve mean", fontsize=24)  # 标题
    ax.set_xlabel("Features num", fontsize=20)  # x轴标签
    ax.set_ylabel("R2", fontsize=20)  # y轴标签
    ax.tick_params(labelsize=16)  # 设置坐标轴轴刻度大小
    ax.legend(loc="best", prop={"size": 16})  # 图例
    plt.savefig("./evaluation/rfe_graph_mean.png")  # 保存图像
    # plt.show()  # 显示图像