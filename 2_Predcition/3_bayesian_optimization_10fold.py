import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_validate


data = pd.read_csv("./data/feature_data/STRUM_Q3421_fea_table_double.csv")  # 读取数据
data = data.drop("ID", axis=1)  # 删除ID列

rfe_infos = pd.read_excel("./resource/rfe_infos.xlsx")  # 读取RFE模型的特征信息
X_cols = rfe_infos[rfe_infos["ranking"] == 1]["feature_names"].tolist()  # 最佳特征组合
X = data[X_cols].values  # 取出特征值 X
y = data["Experimental_DDG"].values  # 取出目标值 y

sc = StandardScaler()  # 定义标准化模型
X = sc.fit_transform(X)  # 标准化



def xgb_cv(n_estimators,max_depth,eta,subsample,colsample_bytree,learning_rate):
    params={
        'n_estimators': 10,
        'max_depth': 1,
        'eta': 0.01,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'learning_rate': 0.1
        }
    params.update({'n_estimators':int(n_estimators),'max_depth':int(max_depth),'eta':eta,'subsample':subsample,'colsample_bytree':colsample_bytree,'learning_rate':learning_rate})
    model=XGBRegressor(**params)
    cv_result=cross_validate(model,X,y,cv=10,scoring='r2',return_train_score=True)
    return cv_result.get('test_score').mean()


param_value_dics={
                   'n_estimators':(10, 1000),
                    'max_depth':(1, 10),
                   'eta':(0.01,1),
                   'subsample':(0.1, 1.0),
                   'colsample_bytree':(0.1, 1.0),
                   'learning_rate': (0.001,0.1)
               }




# 建立贝叶斯调参对象，迭代20次
lgb_bo = BayesianOptimization(
        xgb_cv,
        param_value_dics,
        random_state=42
    )
lgb_bo.maximize(init_points=5,n_iter=100) #init_points-调参基准点，n_iter-迭代次数

# 查看最优参数结果
print(lgb_bo.max)
# 查看全部调参结果
print(lgb_bo.res)

# 在当前调参的结果下，可以再次细化设定或修改参数范围、进一步调参
# lgb_bo.set_bounds({'n_estimators':(400, 450)})
# lgb_bo.maximize(init_points=1,n_iter=20)



best_param = pd.DataFrame(lgb_bo.max['params'], index=['values'])
best_param.to_excel("./resource/BO_Best_Param.xlsx", index=True)










