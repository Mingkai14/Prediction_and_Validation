import pandas as pd
import numpy as np



eval_both = pd.read_excel('./evaluation/evaluate_both.xlsx')
eval_forward = pd.read_excel('./evaluation/evaluate_forward.xlsx')
eval_reverse = pd.read_excel('./evaluation/evaluate_reverse.xlsx')


columns=['inconsistence','sign_correctly_predicted_forward','sign_correctly_predicted_reverse','forward_r','reverse_r']

res=[]
pred_ddg_forward=list(pd.read_excel('./evaluation/pred_ddg_forward.xlsx')['pred_ddg'])
pred_ddg_reverse=list(pd.read_excel('./evaluation/pred_ddg_reverse.xlsx')['pred_ddg'])
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

forward_true_ddg=read_xls_ddg('./data/raw_data/S125_raw_forward.xls')
reverse_true_ddg=read_xls_ddg('./data/raw_data/S125_raw_reverse.xls')

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

res.append(float(eval_forward.loc[eval_forward.iloc[:,0]=='pearson']['prediction_on_S125_forward']))

res.append(float(eval_reverse.loc[eval_reverse.iloc[:,0]=='pearson']['prediction_on_S125_reverse']))


import xlwt
wb=xlwt.Workbook()
ws=wb.add_sheet('sheet1')
for i in range(len(columns)):
    ws.write(0,i,columns[i])
for i in range(len(res)):
    ws.write(1,i,res[i])
wb.save('./evaluation/evaluation_summary.xls')





