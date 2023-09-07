import pandas as pd
import numpy as np

eval_both = pd.read_excel('./evaluation_Ssym/evaluate_both.xlsx')
eval_forward = pd.read_excel('./evaluation_Ssym/evaluate_forward.xlsx')
eval_reverse = pd.read_excel('./evaluation_Ssym/evaluate_reverse.xlsx')


columns=['direct_RMSE','direct_r','reverse_RMSE','reverse_r','r_dr','bias','inconsistence','sign_correctly_predicted_forward','sign_correctly_predicted_reverse']
res=[]

res.append(float(eval_forward.loc[eval_forward.iloc[:,0]=='rmse']['prediction_on_Ssym_forward']))
res.append(float(eval_forward.loc[eval_forward.iloc[:,0]=='pearson']['prediction_on_Ssym_forward']))


res.append(float(eval_reverse.loc[eval_reverse.iloc[:,0]=='rmse']['prediction_on_Ssym_reverse']))
res.append(float(eval_reverse.loc[eval_reverse.iloc[:,0]=='pearson']['prediction_on_Ssym_reverse']))



pred_ddg_forward=list(pd.read_excel('./evaluation_Ssym/pred_ddg_forward.xlsx')['pred_ddg'])
pred_ddg_reverse=list(pd.read_excel('./evaluation_Ssym/pred_ddg_reverse.xlsx')['pred_ddg'])

pred_ddg_forward_ = np.array(pred_ddg_forward).reshape((-1, 1))
pred_ddg_reverse_ = np.array(pred_ddg_reverse).reshape((-1, 1))
yy = np.concatenate([pred_ddg_forward_, pred_ddg_reverse_], -1)
yy = yy.T
corr_matrix = np.corrcoef(yy)
r_dr = corr_matrix[0][1]
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


import xlrd
def read_xls_ddg(name):
    rb = xlrd.open_workbook(name)
    rs=rb.sheet_by_index(0)
    nrow = rs.nrows
    ddg_list=[]
    for i in range(1,nrow):
        ddg_list.append(rs.cell_value(i,3))
    return ddg_list

forward_true_ddg=read_xls_ddg('./data/raw_data/Ssym_forward_raw.xls')
reverse_true_ddg=read_xls_ddg('./data/raw_data/Ssym_reverse_raw.xls')

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
wb.save('./evaluation_Ssym/evaluation_summary.xls')





