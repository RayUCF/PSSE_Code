import pandas as pd
import numpy as np
import xlwt

xls = pd.ExcelFile('dnn-test.xls')
total = np.zeros(235)
for i in range(16262, 17133):
	sys_data = pd.read_excel(xls, sheetname=str(i), header=None)
	sys_data = sys_data.squeeze()
	# sys_data = sys_data.as_matrix()
	# print(total.shape)
	# print(sys_data.shape)
	total = np.vstack((total, sys_data))
total = total[1:,:]
_r, _c = total.shape
file = xlwt.Workbook()
sheet = file.add_sheet('Sheet1')
for r in range(_r):
	for c in range(_c):
		sheet.write(r, c, total[r, c])
file.save('lstm.xls')

