# import matplotlib.pyplot as plt
# import xlrd
# import numpy as np
# import pandas as pd
# from numpy import linalg as LA
# import xlwt
# 
# # data_sheet = xlrd.open_workbook('result.xls')
# # dataset = data_sheet.sheets()[0]
# 
# 
# 
# real = pd.read_excel('./Results/real.xlsx', sheetname='Sheet1', header=None)
# real = real.as_matrix()
# real_angle = real[-288:, 0:13]  # 7000:8000
# real_voltage = real[-288:, 13:]
# 
# h_real_voltage = []
# for i in range(24):
# 	tmp = np.sum(h_real_voltage[i * 5:(i + 1) * 5 - 1])
# 	h_real_voltage.append(1/12*tmp)
# h_real_voltage = np.array(h_real_voltage)
# h_real_voltage = h_real_voltage.squeeze()
# 
# dnn = pd.read_excel('./Results/dnn-test-total.xls', sheetname='Sheet1', header=None)
# dnn = dnn.as_matrix()
# # sys_data = sys_data.T
# dnn_angle =  dnn[:, 0:13]
# dnn_voltage = dnn[:, 13:]
# 
# wls = pd.read_excel('./Results/wls.xlsx', sheetname='Sheet1', header=None)
# wls = wls.as_matrix()
# wls_angle = wls[-288:, 0:13]
# wls_voltage = wls[-288:, 13:]
# 
# 
# 
# wls_real_angle = (wls_angle - real_angle) ** 2
# wls_real_voltage = (wls_voltage - real_voltage) ** 2
# 
# dnn_real_angle = (dnn_angle - real_angle) ** 2
# dnn_real_voltage = (dnn_voltage - real_voltage) ** 2
# norm2_DNN_VA = 1 / dnn_real_angle.shape[1] * np.sqrt(np.sum(dnn_real_angle, axis=1))
# norm2_WLS_VA = 1 / wls_real_angle.shape[1] * np.sqrt(np.sum(wls_real_angle, axis=1))
# norm2_DNN_VM = 1 / dnn_real_voltage.shape[1] * np.sqrt(np.sum(dnn_real_voltage, axis=1))
# norm2_WLS_VM = 1 / wls_real_voltage.shape[1] * np.sqrt(np.sum(wls_real_voltage, axis=1))
# 
# 
# h_norm2_DNN_VA = []
# for i in range(24):
# 	tmp = np.sum(norm2_DNN_VA[i * 5:(i + 1) * 5 - 1])
# 	h_norm2_DNN_VA.append(1/12*tmp)
# h_norm2_DNN_VA = np.array(h_norm2_DNN_VA)
# h_norm2_DNN_VA = h_norm2_DNN_VA.squeeze()
# 
# h_norm2_WLS_VA = []
# for i in range(24):
# 	tmp = np.sum(norm2_WLS_VA[i * 12:(i + 1) * 12])
# 	h_norm2_WLS_VA.append(1/12*tmp)
# h_norm2_WLS_VA = np.array(h_norm2_WLS_VA)
# h_norm2_WLS_VA = h_norm2_WLS_VA.squeeze()
# 
# h_norm2_DNN_VM = []
# for i in range(24):
# 	tmp = np.sum(norm2_DNN_VM[i * 12:(i + 1) * 12 ])
# 	h_norm2_DNN_VM.append(1/12*tmp)
# h_norm2_DNN_VM = np.array(h_norm2_DNN_VM)
# h_norm2_DNN_VM = h_norm2_DNN_VM.squeeze()
# 
# h_norm2_WLS_VM = []
# for i in range(24):
# 	tmp = np.sum(norm2_WLS_VM[i * 12:(i + 1) * 12])
# 	h_norm2_WLS_VM.append(1/12*tmp)
# h_norm2_WLS_VM = np.array(h_norm2_WLS_VM)
# h_norm2_WLS_VM = h_norm2_WLS_VM.squeeze()
# 
# '''
# norm2_DNN_VA, norm2_WLS_VA, norm2_DNN_VM, norm2_WLS_VM = [], [], [], []
# for i in range(100):
# 
# 	norm2_DNN_VA.append(LA.norm(dnn_angle - real_angle))
# 	norm2_WLS_VA.append(LA.norm(wls_angle - real_angle))
# 	norm2_DNN_VM.append(LA.norm(dnn_voltage - real_voltage))
# 	norm2_WLS_VM.append(LA.norm(wls_voltage - real_voltage))
# 
# 	print('norm2_DNN_VA: ', LA.norm(dnn_angle - real_angle))
# 	print('norm2_WLS_VA: ', LA.norm(wls_angle - real_angle))
# 
# 	print('norm2_DNN_VM: ', LA.norm(dnn_voltage - real_voltage))
# 	print('norm2_WLS_VM: ', LA.norm(wls_voltage - real_voltage))
# 	print(50 * '*')
# 
# file = xlwt.Workbook()
# sheet = file.add_sheet('sheet1', cell_overwrite_ok=True)
# for i in range(100):
# 	# print(type(measurement))
# 	sheet.write(i, 0, norm2_DNN_VA[i])
# 	sheet.write(i, 1, norm2_WLS_VA[i])
# 	sheet.write(i, 2, norm2_DNN_VM[i])
# 	sheet.write(i, 3, norm2_WLS_VM[i])
# file.save('Euclidean.xls')
# '''
# 
# '''
# # 设置图例并且设置图例的字体及大小
# font1 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 12,
#          }
# 
# # 设置坐标刻度值的大小以及刻度值的字体
# # plt.tick_params(labelsize=15)
# 
# font2 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 14,
# }
# 
# angle_axis = np.arange(1, 14, 1)
# print('angle')
# p1.plot(angle_axis, dnn_angle, 'x-', color='red', label='DNN', linewidth=1.0)
# p1.plot(angle_axis, wls_angle, '*-', color='green', label='WLS', linewidth=1.0)
# p1.plot(angle_axis, real_angle,  '.-', color='blue', label='Real', linestyle='--')
# p1.set_xlabel('Bus Number', font2)
# p1.set_ylabel('Voltage Angle(degree)', font2)
# p1.set_xticks(angle_axis)
# p1.grid(axis='y')
# p1.legend()
# #
# magnitude_axis = np.arange(1, 15, 1)
# print('magnitude')
# p2.plot(magnitude_axis, dnn_voltage, 'x-', color='red', label='DNN', linewidth=1.0)
# p2.plot(magnitude_axis, wls_voltage, '*-', color='green', label='WLS', linewidth=1.0)
# p2.plot(magnitude_axis, real_voltage,  '.-', color='blue', label='Real', linestyle='--')
# p2.set_xlabel('Bus Number', font2)
# p2.set_ylabel('Voltage Magnitude(pu)', font2)
# p2.set_xticks(magnitude_axis)
# p2.grid(axis='y')
# p2.legend()
# 
# # legend = plt.legend(prop=font1)
# # plt.xticks(x_axis)
# # plt.grid(axis='y')
# plt.show()
# 
# exit()
# '''
# # plt.tick_params(labelsize=15)
# 
# font2 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 14,
# }
# 
# 
# # '''
# plt.figure()
# p1 = plt.subplot()
# # p2 = plt.subplot(122)
# width = 0.3
# ind = np.arange(len(h_norm2_DNN_VA))
# # fig, ax = plt.subplots()
# rects1 = p1.bar(ind, h_norm2_DNN_VA, width, color='r')
# rects2 = p1.bar(ind + width, h_norm2_WLS_VA, width, color='b')
# 
# print('*****************DNN**************')
# print('5min',norm2_DNN_VA[0])
# print('1h',np.sum(norm2_DNN_VA[0:12])/12)
# print('one day',np.sum(norm2_DNN_VA) / len(norm2_DNN_VA))
# print('*****************WLS**************')
# print('5min',norm2_WLS_VA[0])
# print('1h',np.sum(norm2_WLS_VA[0:12])/12)
# print('one day',np.sum(norm2_WLS_VA) / len(norm2_WLS_VA))
# # add some text for labels, title and axes ticks
# p1.set_ylabel('RMSE', font2)
# p1.set_title('IEEE14 voltage angles', font2)
# # ax.set_xticks(ind + width / 2)
# 
# 
# # ind = np.arange(len(h_norm2_DNN_VM))
# # rects3 = p2.bar(ind, h_norm2_DNN_VM, width, color='r')
# # rects4 = p2.bar(ind + width, h_norm2_WLS_VM, width, color='b')
# # print('*****************DNN**************')
# # print('5min',norm2_DNN_VM[0])
# # print('1h',np.sum(norm2_DNN_VM[0:12])/12)
# # print('one day',np.sum(norm2_DNN_VM) / len(norm2_DNN_VM))
# #
# # print('*****************WLS**************')
# # print('5min',norm2_WLS_VM[0])
# # print('1h',np.sum(norm2_WLS_VM[0:12])/12)
# # print('one day',np.sum(norm2_WLS_VM) / len(norm2_WLS_VM))
# #
# # # add some text for labels, title and axes ticks
# # p2.set_ylabel('RMSE', font2)
# # p2.set_title('IEEE14 voltage magnitudes', font2)
# # # ax.set_xticks(ind + width / 2)
# 
# 
# p1.legend((rects1[0], rects2[0]), ('DNN', 'WLS'))
# # p2.legend((rects3[0], rects4[0]), ('DNN', 'WLS'))
# plt.show()
# '''
# 
# #plot msr_rmse
# width = 0.5
# plt.figure()
# p1 = plt.subplot()
# wls_msr_rmse = pd.read_excel('./Results/msr_rmse.xlsx', sheetname='rmse', header=None)
# wls_msr_rmse  = wls_msr_rmse [584:]
# h_wls_msr_rmse = []
# for i in range(24):
# 	tmp = np.sum(wls_msr_rmse[i * 5:(i + 1) * 5 - 1])
# 	h_wls_msr_rmse.append(1/12*tmp)
# h_wls_msr_rmse = np.array(h_wls_msr_rmse)
# h_wls_msr_rmse = h_wls_msr_rmse.squeeze()
# # wls_msr_rmse = wls_msr_rmse.as_matrix()
# print('*****************WLS**************')
# # print('5min',wls_msr_rmse[0])
# # print('1h',np.sum(wls_msr_rmse[0:12])/12)
# # print('one day',np.sum(norm2_WLS_VM) / len(norm2_WLS_VM))
# 
# # p1.plot(wls_msr_rmse)
# 
# 
# 
# dnn_msr_mse = pd.read_excel('./Results/dnn.xls', sheetname='loss', header=None)
# # dnn_msr_mse = dnn_msr_mse.as_matrix()
# dnn_msr_rmse = np.sqrt(dnn_msr_mse)
# dnn_msr_rmse = dnn_msr_rmse[584:]
# 
# h_dnn_msr_rmse = []
# for j in range(24):
# 	tmp = np.sum(dnn_msr_rmse[ j*5:((j + 1)*5 - 1)])
# 	h_dnn_msr_rmse.append(1/12*tmp)
# h_dnn_msr_rmse = np.array(h_dnn_msr_rmse)
# h_dnn_msr_rmse = h_dnn_msr_rmse.squeeze()
# print('*****************DNN**************')
# # print('5min',norm2_WLS_VM[0])
# # print('1h',np.sum(norm2_WLS_VM[0:12])/12)
# # print('one day',np.sum(norm2_WLS_VM) / len(norm2_WLS_VM))
# 
# 
# ind = np.arange(len(h_dnn_msr_rmse))
# rects1 = p1.bar(ind, h_dnn_msr_rmse, width, color='r')
# rects2 = p1.bar(ind + width, h_wls_msr_rmse, width, color='b')
# p1.set_ylabel('RMSE', font2)
# p1.set_title('Measurement RMSE in IEEE14', font2)
# p1.legend((rects1[0], rects2[0]), ('DNN', 'WLS'))
# # plt.plot(ind, dnn_msr_rmse, ind, wls_msr_rmse)
# 
# plt.show()
# 
# '''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from matplotlib.patches import ConnectionPatch
import xlwt

# # 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }

wls = pd.read_excel('./Results/wls.xlsx', sheetname='Sheet1', header=None)
wls = wls.as_matrix()
mask = np.where(np.isnan(wls[:, 0]))[0]
wls = np.delete(wls, mask, axis=0)


wls_angle = wls[:, 0:13]
wls_voltage = wls[:, 13:]

dnn = pd.read_excel('./Results/dnn-test-total.xls', sheetname='Sheet1', header=None)
dnn = dnn.as_matrix()
dnn = np.delete(dnn, mask, axis=0)


# sys_data = sys_data.T
dnn_angle = dnn[:, 0:13]
dnn_voltage = dnn[:, 13:]

real = pd.read_excel('./Results/real.xlsx', sheetname='Sheet1', header=None)
real = real.as_matrix()
real = np.delete(real, mask, axis=0)

real_angle = real[:, 0:13]
real_voltage = real[:, 13:]

wls_real_angle = (wls_angle - real_angle) ** 2
wls_real_voltage = (wls_voltage - real_voltage) ** 2

dnn_real_angle = (dnn_angle - real_angle) ** 2
dnn_real_voltage = (dnn_voltage - real_voltage) ** 2
norm2_DNN_VA = 1 / dnn_real_angle.shape[1] * np.sqrt(np.sum(dnn_real_angle, axis=1))
norm2_WLS_VA = 1 / wls_real_angle.shape[1] * np.sqrt(np.sum(wls_real_angle, axis=1))
mask1 = np.where(norm2_WLS_VA > 0.05)
norm2_DNN_VM = 1 / dnn_real_voltage.shape[1] * np.sqrt(np.sum(dnn_real_voltage, axis=1))
norm2_WLS_VM = 1 / wls_real_voltage.shape[1] * np.sqrt(np.sum(wls_real_voltage, axis=1))
mask2 = np.where(norm2_WLS_VM > 0.02)
mask3 = np.hstack((mask1, mask2))
mask3 = np.unique(mask3)

norm2_DNN_VA = np.delete(norm2_DNN_VA, mask3, axis=0)
norm2_WLS_VA = np.delete(norm2_WLS_VA, mask3, axis=0)
norm2_DNN_VM = np.delete(norm2_DNN_VM, mask3, axis=0)
norm2_WLS_VM = np.delete(norm2_WLS_VM, mask3, axis=0)

norm2_DNN_VA = norm2_DNN_VA[-288:]
norm2_WLS_VA = norm2_WLS_VA[-288:]
norm2_DNN_VM = norm2_DNN_VM[-288:]
norm2_WLS_VM = norm2_WLS_VM[-288:]


print(len(norm2_DNN_VA))

h_norm2_DNN_VA = []
for i in range(len(norm2_DNN_VA)):
	if (i + 1) % 12 == 0:
		tmp = np.sum(norm2_DNN_VA[i + 1 - 12:i + 1])
		h_norm2_DNN_VA.append(1 / 12 * tmp)
		print(i)
	elif (len(norm2_DNN_VA) - i) < 12:
		tmp = np.sum(norm2_DNN_VA[i: len(norm2_DNN_VA)])
		h_norm2_DNN_VA.append(1 / (len(norm2_WLS_VM) - i) * tmp)
		break
h_norm2_DNN_VA = np.array(h_norm2_DNN_VA)
h_norm2_DNN_VA = h_norm2_DNN_VA.squeeze()

h_norm2_WLS_VA = []
for i in range(len(norm2_WLS_VA)):
	if (i + 1) % 12 == 0:
		tmp = np.sum(norm2_WLS_VA[i + 1 - 12:i + 1])
		h_norm2_WLS_VA.append(1 / 12 * tmp)
	elif (len(norm2_WLS_VA) - i - 1) < 12:
		tmp = np.sum(norm2_WLS_VA[i: len(norm2_WLS_VA)])
		h_norm2_WLS_VA.append(1 / (len(norm2_WLS_VA) - i) * tmp)
		break
h_norm2_WLS_VA = np.array(h_norm2_WLS_VA)
h_norm2_WLS_VA = h_norm2_WLS_VA.squeeze()

h_norm2_DNN_VM = []
for i in range(len(norm2_DNN_VM)):
	if (i + 1) % 12 == 0:
		tmp = np.sum(norm2_DNN_VM[i + 1 - 12:i + 1])
		h_norm2_DNN_VM.append(1 / 12 * tmp)
	elif (len(norm2_DNN_VM) - i - 1) < 12:
		tmp = np.sum(norm2_DNN_VM[i: len(norm2_DNN_VM)])
		h_norm2_DNN_VM.append(1 / (len(norm2_DNN_VM) - i) * tmp)
		break
h_norm2_DNN_VM = np.array(h_norm2_DNN_VM)
h_norm2_DNN_VM = h_norm2_DNN_VM.squeeze()

h_norm2_WLS_VM = []
for i in range(len(norm2_WLS_VM)):
	if (i + 1) % 12 == 0:
		tmp = np.sum(norm2_WLS_VM[i + 1 - 12:i + 1])
		h_norm2_WLS_VM.append(1 / 12 * tmp)
	elif (len(norm2_WLS_VM) - i - 1) < 12:
		tmp = np.sum(norm2_WLS_VM[i: len(norm2_WLS_VM)])
		h_norm2_WLS_VM.append(1 / (len(norm2_WLS_VM) - i) * tmp)
		break
h_norm2_WLS_VM = np.array(h_norm2_WLS_VM)
h_norm2_WLS_VM = h_norm2_WLS_VM.squeeze()

# print('*****************DNN**************')
# print('5min', norm2_DNN_VA[0])
# print('1h', np.sum(norm2_DNN_VA[0:12]) / 12)
# print('one day', np.sum(norm2_DNN_VA) / len(norm2_DNN_VA))
# print('*****************WLS**************')
# print('5min', norm2_WLS_VA[0])
# print('1h', np.sum(norm2_WLS_VA[0:12]) / 12)
# print('one day', np.sum(norm2_WLS_VA) / len(norm2_WLS_VA))
#
# print('*****************DNN**************')
# print('5min', norm2_DNN_VM[0])
# print('1h', np.sum(norm2_DNN_VM[0:12]) / 12)
# print('one day', np.sum(norm2_DNN_VM) / len(norm2_DNN_VM))
#
# print('*****************WLS**************')
# print('5min', norm2_WLS_VM[0])
# print('1h', np.sum(norm2_WLS_VM[0:12]) / 12)
# print('one day', np.sum(norm2_WLS_VM) / 280)


# print('DNN_VA5MIN', np.max(norm2_DNN_VA))
# print('WLS_VA5MIN', np.max(norm2_WLS_VA))
# print('DNN_VM5MIN', np.max(norm2_DNN_VM))
# print('WLS_VM5MIN', np.max(norm2_WLS_VM))
#
# print('DNN_VA1H', np.max(h_norm2_DNN_VA))
# print('WLS_VA1H', np.max(h_norm2_WLS_VA))
# print('DNN_VM1H', np.max(h_norm2_DNN_VM))
# print('WLS_VM1H', np.max(h_norm2_WLS_VM))

print('DNN_VA5MIN', np.min(norm2_DNN_VA))
print('WLS_VA5MIN', np.min(norm2_WLS_VA))
print('DNN_VM5MIN', np.min(norm2_DNN_VM))
print('WLS_VM5MIN', np.min(norm2_WLS_VM))

print('DNN_VA1H', np.min(h_norm2_DNN_VA))
print('WLS_VA1H', np.min(h_norm2_WLS_VA))
print('DNN_VM1H', np.min(h_norm2_DNN_VM))
print('WLS_VM1H', np.min(h_norm2_WLS_VM))





# print(np.sum(norm2_WLS_VA))
plt.figure()
p2 = plt.subplot()
# p2 = plt.subplot(122)
width = 0.3


# ind = np.arange(len(h_norm2_DNN_VA))
# # fig, ax = plt.subplots()
# rects1 = p1.bar(ind, h_norm2_DNN_VA, width, color='r')
# rects2 = p1.bar(ind + width, h_norm2_WLS_VA, width, color='b')
# # add some text for labels, title and axes ticks
# p1.set_ylabel('THRMSE', font2)
# p1.set_title('IEEE14 voltage angles',font2)
# # ax.set_xticks(ind + width / 2)
#
# p1.legend((rects1[0], rects2[0]), ('DNN', 'WLS'))

ind = np.arange(len(norm2_DNN_VM))
rects3 = p2.bar(ind, norm2_DNN_VM, width, color='r')
rects4 = p2.bar(ind + width, norm2_WLS_VM, width, color='b')
# add some text for labels, title and axes ticks
p2.set_ylabel('THRMSE', font2)
p2.set_title('IEEE14 voltage magnitudes', font2)
p2.legend((rects3[0], rects4[0]), ('DNN', 'WLS'))


plt.tick_params(labelsize=15)
plt.show()



