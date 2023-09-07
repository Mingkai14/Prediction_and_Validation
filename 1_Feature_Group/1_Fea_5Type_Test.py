import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.model_selection import KFold


# 平均值随着类别数的增多是上升的
# a类特征与别人的组合还是有用的
# 看每一个单独特征对别的特征组合有没有帮助，例如拿bc比abc，bcd比abcd
#单独的特征类别排序
#全部的特征排序，最佳的特征组合类，看过滤后的特征类是不是也是一致的
#确实是c类最牛

# I want to prove more type performance is better
# I want to prove a,b,c,d,e, all have positive effect
# I want to prove c class has most important effect
# I want to prove despite single a class





#load data and drop ID column
df = pd.read_csv('./data/feature_data/STRUM_Q3421_fea_table_double.csv')
df.drop('ID', axis=1, inplace=True)

fea_index_class_a=[0,1]
fea_index_class_b=[2,651]
fea_index_class_c=[652,1279]
fea_index_class_d=[1280,1425]
fea_index_class_e=[1426,1451]
#whole 1408 features
names_class_a=[]
names_class_b=[]
names_class_c=[]
names_class_d=[]
names_class_e=[]

table_names = list(df.columns)

for i in range(fea_index_class_a[0],fea_index_class_a[1]+1):
    names_class_a.append(table_names[i])
for i in range(fea_index_class_b[0],fea_index_class_b[1]+1):
    names_class_b.append(table_names[i])
for i in range(fea_index_class_c[0],fea_index_class_c[1]+1):
    names_class_c.append(table_names[i])
for i in range(fea_index_class_d[0],fea_index_class_d[1]+1):
    names_class_d.append(table_names[i])
for i in range(fea_index_class_e[0],fea_index_class_e[1]+1):
    names_class_e.append(table_names[i])


basic_class_dict={'a':names_class_a,'b':names_class_b,'c':names_class_c,'d':names_class_d,'e':names_class_e}
class_dict={}

#layer1
for key in basic_class_dict.keys():
    class_dict[key]=basic_class_dict[key]
#layer2
for key1 in basic_class_dict.keys():
    for key2 in basic_class_dict.keys():
        order_list=sorted([str(key1),str(key2)])
        temp_key=''.join(order_list)
        if key1==key2 or temp_key in class_dict.keys():
            continue
        temp_list=basic_class_dict[key1]+basic_class_dict[key2]
        class_dict[temp_key]=temp_list
#layer3
for key1 in basic_class_dict.keys():
    for key2 in basic_class_dict.keys():
        for key3 in basic_class_dict.keys():
            order_list = sorted([str(key1), str(key2),str(key3)])
            temp_key = ''.join(order_list)
            if key1==key2 or key1==key3 or key2==key3 or temp_key in class_dict.keys():
                continue
            temp_list=basic_class_dict[key1]+basic_class_dict[key2]+basic_class_dict[key3]
            class_dict[temp_key]=temp_list
#layer4
for key1 in basic_class_dict.keys():
    for key2 in basic_class_dict.keys():
        for key3 in basic_class_dict.keys():
            for key4 in basic_class_dict.keys():
                order_list = sorted([str(key1), str(key2), str(key3),str(key4)])
                temp_key = ''.join(order_list)
                if key1==key2 or key1==key3 or key1==key4 or key2==key3 or key2==key4 or key3==key4 or temp_key in class_dict.keys():
                    continue
                temp_list=basic_class_dict[key1]+basic_class_dict[key2]+basic_class_dict[key3]+basic_class_dict[key4]
                class_dict[temp_key]=temp_list
#layer5
for key1 in basic_class_dict.keys():
    for key2 in basic_class_dict.keys():
        for key3 in basic_class_dict.keys():
            for key4 in basic_class_dict.keys():
                for key5 in basic_class_dict.keys():
                    order_list = sorted([str(key1), str(key2), str(key3),str(key4),str(key5)])
                    temp_key = ''.join(order_list)
                    if key1==key2 or key1==key3 or key1==key4 or key1==key5 or key2==key3 or key2==key4 or key2==key5 or key3==key4 or key3==key5 or key4==key5 or temp_key in class_dict.keys():
                        continue
                    temp_list=basic_class_dict[key1]+basic_class_dict[key2]+basic_class_dict[key3]+basic_class_dict[key4]
                    class_dict[temp_key]=temp_list


def Compute(df,table_names:list,order:str):
    table_names = table_names + ['Experimental_DDG', 'Experimental_DDG_Classification']
    df = df.loc[:, table_names]

    Y_reg = list(df['Experimental_DDG'])
    Y_cls = list(df['Experimental_DDG_Classification'])

    df.drop(['Experimental_DDG', 'Experimental_DDG_Classification'], axis=1, inplace=True)
    fea = df.values

    # 数值型特征-标准化
    sc = StandardScaler()
    fea = sc.fit_transform(fea)

    X = np.array(fea)
    y = np.array(Y_reg)

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    count = 0
    mse_list = []
    rmse_list = []
    mae_list = []
    r2_list = []
    pearson_list = []
    spearman_list = []

    for train_idx, test_idx in kfold.split(X):
        count += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # create model and train
        model = XGBRegressor()

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

        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        pearson_list.append(pearson)
        spearman_list.append(spearman)

    mean_mse=sum(mse_list)/10
    mean_rmse=sum(rmse_list)/10
    mean_mae=sum(mae_list)/10
    mean_r2=sum(r2_list)/10
    mean_pearson=sum(pearson_list)/10
    mean_spearman=sum(spearman_list)/10

    return [order,mean_r2]

res_dict={}
for key in class_dict:
    res=Compute(df,class_dict[key],str(key))
    res_dict[res[0]]=res[1]

print(res_dict)

from matplotlib import pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 9), dpi=100)  # 定义画布
fig.autofmt_xdate()
ax.plot(
    list(res_dict.keys()),
    list(res_dict.values()),
    marker="",
    linestyle="-",
    linewidth=2,
    label=f"features group",
)  # 各交叉验证均值结果曲线
for i,j in zip(list(res_dict.keys()),list(res_dict.values())):
    ax.text(i,j,str(round(j,3)))
ax.set_title("Feature groups screening curve", fontsize=24)  # 标题
ax.set_xlabel("Features group", fontsize=10)  # x轴标签
ax.set_ylabel("R2", fontsize=16)  # y轴标签
ax.tick_params(labelsize=10)  # 设置坐标轴轴刻度大小
ax.legend(loc="best", prop={"size": 16})  # 图例
plt.savefig("./evaluation/5Type_Combination_Curve.png")  # 保存图像
plt.show()  # 显示图像

total_dict={}
count_dict={}
mean_dict={}
for key in res_dict.keys():
    s_len= len(str(key))
    if 'layer'+str(s_len) not in total_dict.keys():
        total_dict['layer'+str(s_len)]=0
    total_dict['layer'+str(s_len)]+=res_dict[key]
    if 'layer'+str(s_len) not in count_dict.keys():
        count_dict['layer'+str(s_len)]=0
    count_dict['layer'+str(s_len)]+=1
for key in total_dict.keys():
    mean_dict[key]=total_dict[key]/count_dict[key]
print(mean_dict)



from matplotlib import pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 9), dpi=100)  # 定义画布
fig.autofmt_xdate()
ax.plot(
    list(mean_dict.keys()),
    list(mean_dict.values()),
    marker="",
    linestyle="-",
    linewidth=2,
    label=f"features layer",
)  # 各交叉验证均值结果曲线
for i,j in zip(list(mean_dict.keys()),list(mean_dict.values())):
    ax.text(i,j,str(round(j,3)))
ax.set_title("Feature layers screening curve", fontsize=24)  # 标题
ax.set_xlabel("Features group", fontsize=10)  # x轴标签
ax.set_ylabel("R2", fontsize=16)  # y轴标签
ax.tick_params(labelsize=10)  # 设置坐标轴轴刻度大小
ax.legend(loc="best", prop={"size": 16})  # 图例
plt.savefig("./evaluation/5Layer_Mean_Performance_Curve.png")  # 保存图像
plt.show()  # 显示图像



def Judge(order:str,res_dict:dict):
    return_dict={}
    for key in res_dict.keys():
        if order not in str(key):
            type_without_order=str(key)
            type_with_type=type_without_order+order
            foo_list=sorted(list(type_with_type))
            type_with_type=''.join(foo_list)
            return_dict[type_without_order+'_'+type_with_type]=res_dict[type_with_type]-res_dict[type_without_order]
    total=0
    count=0
    for key in return_dict:
        count+=1
        total+=return_dict[key]
    return [return_dict,total/count]

affect_dict={}
res=Judge('a',res_dict)
print(res[0])
affect_dict['a']=res[1]
res=Judge('b',res_dict)
print(res[0])
affect_dict['b']=res[1]
res=Judge('c',res_dict)
print(res[0])
affect_dict['c']=res[1]
res=Judge('d',res_dict)
print(res[0])
affect_dict['d']=res[1]
res=Judge('e',res_dict)
print(res[0])
affect_dict['e']=res[1]


from matplotlib import pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 9), dpi=100)  # 定义画布
fig.autofmt_xdate()
ax.bar(
    list(affect_dict.keys()),
    list(affect_dict.values()),
    color='blue',
    label=f"features layer",
    width=0.5
)  # 各交叉验证均值结果曲线
for i,j in zip(list(affect_dict.keys()),list(affect_dict.values())):
    ax.text(i,j,str(round(j,3)))
ax.set_title("Feature layers screening curve", fontsize=24)  # 标题
ax.set_xlabel("Features group", fontsize=10)  # x轴标签
ax.set_ylabel("Effect", fontsize=16)  # y轴标签
ax.tick_params(labelsize=10)  # 设置坐标轴轴刻度大小
ax.legend(loc="best", prop={"size": 16})  # 图例
plt.savefig("./evaluation/5Type_Mean_Effect.png")  # 保存图像
plt.show()  # 显示图像
