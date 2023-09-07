import pandas as pd
import numpy as np

eval_both = pd.read_excel('./evaluation_S613/evaluate_both.xlsx')
eval_forward = pd.read_excel('./evaluation_S613/evaluate_forward.xlsx')
eval_reverse = pd.read_excel('./evaluation_S613/evaluate_reverse.xlsx')


columns=['total_r','total_RMSE','total_MAE','direct_r','direct_RMSE','direct_MAE','reverse_r','reverse_RMSE','reverse_MAE','r_dr','bias','inconsistence','sign_correctly_predicted_forward','sign_correctly_predicted_reverse']
res=[]
res.append(float(eval_both.loc[eval_both.iloc[:,0]=='pearson']['prediction_on_S613_both']))
res.append(float(eval_both.loc[eval_both.iloc[:,0]=='rmse']['prediction_on_S613_both']))
res.append(float(eval_both.loc[eval_both.iloc[:,0]=='mae']['prediction_on_S613_both']))

res.append(float(eval_forward.loc[eval_forward.iloc[:,0]=='pearson']['prediction_on_S613_forward']))
res.append(float(eval_forward.loc[eval_forward.iloc[:,0]=='rmse']['prediction_on_S613_forward']))
res.append(float(eval_forward.loc[eval_forward.iloc[:,0]=='mae']['prediction_on_S613_forward']))

res.append(float(eval_reverse.loc[eval_reverse.iloc[:,0]=='pearson']['prediction_on_S613_reverse']))
res.append(float(eval_reverse.loc[eval_reverse.iloc[:,0]=='rmse']['prediction_on_S613_reverse']))
res.append(float(eval_reverse.loc[eval_reverse.iloc[:,0]=='mae']['prediction_on_S613_reverse']))



pred_ddg_forward=list(pd.read_excel('./evaluation_S613/pred_ddg_forward.xlsx')['pred_ddg'])
pred_ddg_reverse=list(pd.read_excel('./evaluation_S613/pred_ddg_reverse.xlsx')['pred_ddg'])
covariance = np.cov(pred_ddg_forward, pred_ddg_reverse)[0, 1]
std_deviation_forward = np.std(pred_ddg_forward)
std_deviation_reverse = np.std(pred_ddg_reverse)
r_dr=covariance/(std_deviation_forward*std_deviation_reverse)
res.append(r_dr)


assert len(pred_ddg_forward)==len(pred_ddg_reverse)
count=len(pred_ddg_forward)
sum=0.0
for i in range(count):
    sum+=pred_ddg_forward[i]+pred_ddg_reverse[i]
bias=sum/(count*2)
res.append(bias)


assert len(pred_ddg_forward)==len(pred_ddg_reverse)
count=len(pred_ddg_forward)
inconsistent_count=0
for i in range(count):
    if pred_ddg_forward[i]*pred_ddg_reverse[i]>0:
        inconsistent_count+=1
inconsistence=inconsistent_count/count

res.append(inconsistence)

deletion=['3DV0_I_V129A','3DV0_I_V129G']
import xlrd
def read_xls_ddg(name):
    rb = xlrd.open_workbook(name)
    rs=rb.sheet_by_index(0)
    nrow = rs.nrows
    ddg_list_f=[]
    ddg_list_r=[]
    for i in range(1,nrow):
        pdb=rs.cell_value(i,0)
        if isinstance(rs.cell_value(i,2),float):
            chain=str(int(rs.cell_value(i,2)))
        else:
            chain=rs.cell_value(i,2)
        mut=rs.cell_value(i,1)
        id=pdb+'_'+chain+'_'+mut
        if id not in deletion:
            ddg_list_f.append(rs.cell_value(i,3))
            ddg_list_r.append(-float(rs.cell_value(i,3)))
    return ddg_list_f,ddg_list_r

forward_true_ddg,reverse_true_ddg=read_xls_ddg('./data/raw_data/S613_raw.xls')

assert len(forward_true_ddg)==len(pred_ddg_forward)
assert len(reverse_true_ddg)==len(pred_ddg_reverse)

sign_correctly_prediction_forward=0
count=len(forward_true_ddg)
for i in range(count):
    if forward_true_ddg[i]*pred_ddg_forward[i]>0:
        sign_correctly_prediction_forward+=1
sign_correctly_predicted_forward=sign_correctly_prediction_forward/count

res.append(sign_correctly_predicted_forward)

sign_correctly_prediction_reverse=0
count=len(reverse_true_ddg)
for i in range(count):
    if reverse_true_ddg[i]*pred_ddg_reverse[i]>0:
        sign_correctly_prediction_reverse+=1
sign_correctly_predicted_reverse=sign_correctly_prediction_reverse/count

res.append(sign_correctly_predicted_reverse)

import xlwt
wb=xlwt.Workbook()
ws=wb.add_sheet('sheet1')
for i in range(len(columns)):
    ws.write(0,i,columns[i])
for i in range(len(res)):
    ws.write(1,i,res[i])
wb.save('./evaluation_S613/evaluation_summary.xls')





