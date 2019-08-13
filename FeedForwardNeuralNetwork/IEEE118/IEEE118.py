#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import xlrd
import xlwt
from torch.autograd import Variable
from tqdm import tqdm
from tqdm import trange
from cyclic_lr_scheduler import CyclicLR
import torch.nn.init as init
from pandas import ExcelWriter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu"  )
print(torch.cuda.get_device_name(torch.cuda.current_device()))

xls = pd.ExcelFile('measurements.xlsx')
sys_data = pd.read_excel(xls, sheetname='Data', header=None)
sys_data = sys_data.as_matrix()

G = pd.read_excel(xls, sheetname='Real', header=None)
G = G.as_matrix()
# Imag part
imag = pd.read_excel(xls, sheetname='Imag', header=None)
B = imag.as_matrix()

# According to the data frame (np. ndarray)
m_type = sys_data[1]  # type of the measurements
FB = sys_data[2]  # from bus
TB = sys_data[3]  # to bus
R = sys_data[4]  # covariance

vi = np.argwhere(m_type == 1)  # Type1: V
Vmask = FB[vi] - 1
ppi = np.argwhere(m_type == 2).squeeze()  # Type2: Pi
BP_mask = Variable(torch.from_numpy(FB[ppi] - 1).long()).squeeze().to(device)
qi = np.argwhere(m_type == 3).squeeze()  # Type3: Qi
BQ_mask = Variable(torch.from_numpy(FB[qi] - 1).long()).squeeze().to(device)
ppf = np.argwhere(m_type == 4).squeeze()  # Type4: Pij
PF_FR = Variable(torch.from_numpy(FB[ppf] - 1).long()).squeeze().to(device)
PF_TO = Variable(torch.from_numpy(TB[ppf] - 1).long()).squeeze().to(device)
qf = np.argwhere(m_type == 5).squeeze()  # Type5: Qij
QF_FR = Variable(torch.from_numpy(FB[qf] - 1).long()).squeeze().to(device)
QF_TO = Variable(torch.from_numpy(TB[qf] - 1).long()).squeeze().to(device)

nvi = len(vi)
npi = len(ppi)
nqi = len(qi)
npf = len(ppf)
nqf = len(qf)

# Load features of the measurements
nBus = 118  # Bus # of targeted system
M = sys_data.shape[1]  # Measurements #
nSet = sys_data.shape[0]  # Sets # of sys_data
N = 2 * nBus - 1  # State variables #
ref_ang = 30 * math.pi / 180  # reference bus angle

# Use Variable container
G = Variable(torch.from_numpy(G).float()).to(device)
B = Variable(torch.from_numpy(B).float()).to(device)

# Mini-batch DataLoader
# Hyper parameters
BATCH_SIZE = 32
LR = 0.001
EPOCH = 300

iter_size = (len(sys_data)-5) // BATCH_SIZE
sys_data1 = torch.from_numpy(sys_data[5:BATCH_SIZE*iter_size+5,:]).float()
torch_dataset1 = Data.TensorDataset(sys_data1, sys_data1)
loader1 = Data.DataLoader(dataset=torch_dataset1,  # torch TensorDataset format
                          batch_size=BATCH_SIZE,  # mini batch size
                          shuffle=False,  # random shuffle for training
                          num_workers=1,  # sub-processes for loading data
                          drop_last=True, )


sys_data2 = torch.from_numpy(sys_data[5:]).float()

true_data = pd.read_excel(xls, sheetname='WLS_Train', header=None)
true_data = true_data.as_matrix()
true_data = torch.from_numpy(true_data).float()

torch_dataset2 = Data.TensorDataset(sys_data2, true_data)
loader2 = Data.DataLoader(dataset=torch_dataset2,  # torch TensorDataset format
                         batch_size=BATCH_SIZE,  # mini batch size
                         shuffle=False,  # random shuffle for training
                         num_workers=2,  # sub-processes for loading data
                         drop_last=True,
                         )



# Build up the NN module
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.hidden1 = torch.nn.Linear(M, 512)
		self.hidden2 = torch.nn.Linear(512, 512)
		self.hidden3 = torch.nn.Linear(512, 512)
		self.hidden4 = torch.nn.Linear(512, 512)
		self.hidden5 = torch.nn.Linear(512, 512)
		# self.hidden3 = torch.nn.Linear(1024, 1024)
		self.output = torch.nn.Linear(512, N)
	
	def forward(self, x):
		x = torch.tanh(self.hidden1(x))
		x = torch.tanh(self.hidden2(x))  # replace sigmoid(), cz it is too slow here
		x = torch.tanh(self.hidden3(x))
		x = torch.tanh(self.hidden4(x))
		x = torch.tanh(self.hidden5(x))
		# x = torch.sigmoid(self.hidden3(x))
		x = self.output(x)
		return x


def weight_init(m):
	'''
	Usage:
		model = Model()
		model.apply(weight_init)
	'''
	if isinstance(m, nn.Conv1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.BatchNorm1d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm2d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm3d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.Linear):
		init.xavier_normal_(m.weight.data)
		init.normal_(m.bias.data)
	elif isinstance(m, nn.LSTM):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.LSTMCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRU):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRUCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)


# Reload the module after initialization
estimator = Net()
print(estimator)
estimator.to(device)
optimizer = torch.optim.Adam(estimator.parameters(), lr=LR, betas=(0.9, 0.9))
loss_func = torch.nn.MSELoss(reduction='mean').to(device)
l2 = torch.nn.MSELoss(reduction='sum').to(device)

# Power Injection
GP = G[BP_mask]
BP = B[BP_mask]
GQ = G[BQ_mask]
BQ = B[BQ_mask]
# Power Flow
GPF = G[PF_FR, PF_TO].unsqueeze(1)
BPF = B[PF_FR, PF_TO].unsqueeze(1)
GQF = G[QF_FR, QF_TO].unsqueeze(1)
BQF = B[QF_FR, QF_TO].unsqueeze(1)


def power_flow(outputs):
	outputs = outputs.squeeze(0)
	outputs = outputs.permute(1, 0)
	Z_hat = Variable(torch.zeros(M, 1)).to(device)
	A = Variable(ref_ang * torch.ones(nBus, 1)).to(device)
	A[:68] = outputs[0:68]  # voltage angle
	A[69:] = outputs[68:nBus - 1]  # voltage angle
	V = outputs[nBus - 1:]  # voltage magnitude
	
	## V
	Z_hat[vi, :] = V[Vmask]
	
	## Pi
	VP_FR = V[BP_mask].repeat(1, nBus)
	VP_TO = V.repeat(1, nBus).permute(1, 0)[BP_mask]
	
	AP_FR = A[BP_mask].repeat(1, nBus)
	AP_TO = A.repeat(1, nBus).permute(1, 0)[BP_mask]
	
	temp = VP_FR * VP_TO * (GP * torch.cos(AP_FR - AP_TO) + BP * torch.sin(AP_FR - AP_TO))
	Z_hat[ppi, :] = torch.sum(temp, 1, keepdim=True)
	
	## Qi
	VQ_FR = V[BQ_mask].repeat(1, nBus)
	VQ_TO = V.repeat(1, nBus).permute(1, 0)[BQ_mask]
	
	AQ_FR = A[BQ_mask].repeat(1, nBus)
	AQ_TO = A.repeat(1, nBus).permute(1, 0)[BQ_mask]
	
	temp = VQ_FR * VQ_TO * (GQ * torch.sin(AQ_FR - AQ_TO) - BQ * torch.cos(AQ_FR - AQ_TO))
	Z_hat[qi, :] = torch.sum(temp, 1, keepdim=True)
	
	## Pij
	Z_hat[ppf, :] = -V[PF_FR] * V[PF_FR] * GPF - V[PF_FR] * V[PF_TO] * (
			-GPF * torch.cos(A[PF_FR] - A[PF_TO]) - BPF * torch.sin(A[PF_FR] - A[PF_TO]))
	
	# Qij
	Z_hat[qf, :] = V[QF_FR] * V[QF_FR] * BQF - V[QF_FR] * V[QF_TO] * (
			-GQF * torch.sin(A[QF_FR] - A[QF_TO]) + BQF * torch.cos(A[QF_FR] - A[QF_TO]))
	
	# return Z_hat.permute(1, 0)
	return Z_hat.squeeze()


########################################
##Batchsize, Build up 3D operation
########################################
# Power Injection
B_GP = G[BP_mask].repeat(BATCH_SIZE, 1, 1)
B_BP = B[BP_mask].repeat(BATCH_SIZE, 1, 1)
B_GQ = G[BQ_mask].repeat(BATCH_SIZE, 1, 1)
B_BQ = B[BQ_mask].repeat(BATCH_SIZE, 1, 1)
# Power Flow
B_GPF = G[PF_FR, PF_TO].unsqueeze(1).repeat(BATCH_SIZE, 1, 1)
B_BPF = B[PF_FR, PF_TO].unsqueeze(1).repeat(BATCH_SIZE, 1, 1)
B_GQF = G[QF_FR, QF_TO].unsqueeze(1).repeat(BATCH_SIZE, 1, 1)
B_BQF = B[QF_FR, QF_TO].unsqueeze(1).repeat(BATCH_SIZE, 1, 1)


def power_flow_batch(outputs):
	outputs = outputs.permute(1,2,0)
	Z_hat = Variable(torch.zeros(BATCH_SIZE, M, 1)).to(device)
	A = Variable(ref_ang * torch.ones(BATCH_SIZE, nBus, 1)).to(device)
	A[:, :68] = outputs[:, 0:68]  # voltage angle
	A[:, 69:] = outputs[:, 68:nBus - 1]  # voltage angle
	V = outputs[:, nBus - 1:]  # voltage magnitude
	
	## V
	Z_hat[:, vi, :] = V[:, Vmask, :]
	## Pi
	B_VP_FR = V[:, BP_mask].repeat(1, 1, nBus)
	B_VP_TO = V.repeat(1, 1, nBus).permute(0, 2, 1)[:, BP_mask]
	
	B_AP_FR = A[:, BP_mask].repeat(1, 1, nBus)
	B_AP_TO = A.repeat(1, 1, nBus).permute(0, 2, 1)[:, BP_mask]
	
	temp = B_VP_FR * B_VP_TO * (B_GP * torch.cos(B_AP_FR - B_AP_TO) + B_BP * torch.sin(B_AP_FR - B_AP_TO))
	Z_hat[:, ppi, :] = torch.sum(temp, 2, keepdim=True)
	
	## Qi
	B_VQ_FR = V[:, BQ_mask].repeat(1, 1, nBus)
	B_VQ_TO = V.repeat(1, 1, nBus).permute(0, 2, 1)[:, BQ_mask]
	
	B_AQ_FR = A[:, BQ_mask].repeat(1, 1, nBus)
	B_AQ_TO = A.repeat(1, 1, nBus).permute(0, 2, 1)[:, BQ_mask]
	
	temp = B_VQ_FR * B_VQ_TO * (B_GQ * torch.sin(B_AQ_FR - B_AQ_TO) - B_BQ * torch.cos(B_AQ_FR - B_AQ_TO))
	Z_hat[:, qi, :] = torch.sum(temp, 2, keepdim=True)
	## Pij
	Z_hat[:, ppf, :] = -V[:, PF_FR] * V[:, PF_FR] * B_GPF - V[:, PF_FR] * V[:, PF_TO] * (
			-B_GPF * torch.cos(A[:, PF_FR] - A[:, PF_TO]) - B_BPF * torch.sin(A[:, PF_FR] - A[:, PF_TO]))
	
	# Qij
	Z_hat[:, qf, :] = V[:, QF_FR] * V[:, QF_FR] * B_BQF - V[:, QF_FR] * V[:, QF_TO] * (
			-B_GQF * torch.sin(A[:, QF_FR] - A[:, QF_TO]) + B_BQF * torch.cos(A[:, QF_FR] - A[:, QF_TO]))
	
	return Z_hat.squeeze(2)


def batch_train_params():
	print('Start Initializing...')
	estimator.apply(weight_init)
	scheduler = CyclicLR(optimizer, base_lr=1e-7, max_lr=1e-3, step_size=300, mode='triangular')
	with trange(0, 50) as numEpoch:
		for epoch in numEpoch:
			scheduler.step()
			for step, (batch_train, batch_target) in enumerate(loader2):
				batch_train = batch_train.view(1, batch_train.shape[0], batch_train.shape[1])
				inputs = Variable(batch_train, requires_grad=True).to(device)
				targets = Variable(batch_target).to(device)
				outputs= estimator(inputs)
				loss = loss_func(outputs, targets)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# numEpoch.set_postfix(Loss='%0.32f' % loss)
			print('Loss=%0.8f' % loss)
	torch.save(estimator.state_dict(), 'initial_batch_params.pkl')
	return loss

# def batch_train_params():
# 	estimator.apply(weight_init)
# 	# Flat start to initialize the parameters of module
# 	init_sv = torch.ones(N)
# 	init_sv[:nBus - 1] = ref_ang * init_sv[:nBus - 1]
# 	init_sv = init_sv.repeat(BATCH_SIZE, 1).to(device)
#
# 	with trange(0, 200) as numEpoch:
# 		for epoch in numEpoch:
# 			# scheduler.step()
# 			for step, (batch_train, _) in enumerate(loader):
# 				batch_train = batch_train.view(1, batch_train.shape[0],
# 				                               batch_train.shape[1])  # (seq_len, batch, input_size)
# 				# init_sv = init_sv.view(1, init_sv.shape[0], init_sv.shape[1])
# 				inputs = Variable(batch_train, requires_grad=True).to(device)
# 				targets = Variable(init_sv).to(device)
#
# 				def closure():
# 					# zero out the gradients
# 					optimizer.zero_grad()
# 					# get the output sequence from the input and the initial hidden and cell states
# 					output = estimator(inputs)
# 					# calculate the loss
# 					loss = loss_func(output, targets)
# 					# print(loss.item())
# 					# calculate the gradients
# 					loss.backward()
# 					return loss
#
# 				epochloss = optimizer.step(closure)
# 			numEpoch.set_postfix(Loss='%0.16f' % epochloss)
# 	torch.save(estimator.state_dict(), 'initial_batch_params.pkl')


def train():
	print('Training...')
	estimator.load_state_dict(torch.load('initial_batch_params.pkl'))
	scheduler = CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-5, step_size=300, mode='triangular')
	
	with trange(0, EPOCH) as numEPoch:
		for epoch in numEPoch:
			total_loss = 0
			scheduler.step()
			optimizer.zero_grad()
			for step, (batch_train, batch_targets) in enumerate(loader1):
				batch_train = batch_train.view(1, batch_train.shape[0], batch_train.shape[1])
				inputs = Variable(batch_train, requires_grad=True).to(device)
				targets = Variable(batch_targets).to(device)
				
				# optimizer.zero_grad()
				# get the output sequence from the input and the initial hidden and cell states
				outputs = estimator(inputs)
				Z_hat = power_flow_batch(outputs)
				# calculate the loss
				loss = loss_func(Z_hat, targets)/iter_size
				# calculate the gradients
				loss.backward()
				total_loss += loss
			optimizer.step()
			
			# def closure():
			# 	# zero out the gradients
			# 	optimizer.zero_grad()
			# 	# get the output sequence from the input and the initial hidden and cell states
			# 	outputs = estimator(inputs)
			# 	Z_hat = power_flow_batch(outputs)
			# 	# calculate the loss
			# 	loss = loss_func(Z_hat, targets)
			# 	# print(loss.item())
			# 	# calculate the gradients
			# 	loss.backward()
			# 	return loss
			# epochloss = optimizer.step(closure)
			numEPoch.set_postfix(Loss='%0.8f' % total_loss)  # print error
			# print('Loss=%0.8f' % total_loss)
	torch.save(estimator.state_dict(), 'final_params.pkl')
	return loss


# Load streaming data
stream_data = pd.read_excel(xls, sheetname='TestData', header=None)
stream_data = stream_data.as_matrix()
# stream_data = stream_data.T
stream_data = torch.from_numpy(stream_data).float()
stream_torch_dataset = Data.TensorDataset(stream_data, stream_data)
stream_loader = Data.DataLoader(dataset=stream_torch_dataset, batch_size=1, shuffle=False,
                                # random shuffle for training
                                num_workers=2,  # sub-processes for loading data
                                )


def pfm_test():
	###########################################################
	weights_SP = []
	with torch.no_grad():
		for param in estimator.parameters():
			weights_SP.append(param.clone())
	###########################################################
	
	for param_group in optimizer.param_groups:
		param_group['lr'] = 1e-8
	# estimator.load_state_dict(torch.load('final_params.pkl'))
	# print(estimator)
	print('Online Training...')
	file = xlwt.Workbook()
	# his_loss = []
	for step, (batch_train, batch_targets) in enumerate(stream_loader):
		inputs = batch_train.view((1, batch_train.shape[0], batch_train.shape[1])).to(device).requires_grad_()
		targets = batch_targets.to(device)
		targets = targets.squeeze()
		
		outputs = estimator(inputs)
		
		Z_hat = power_flow(outputs)
		Z_hat = Z_hat.squeeze()
		loss = loss_func(Z_hat, targets)
		
		# norm2 = 0
		# for n, param in enumerate(estimator.parameters()):
		# 	norm2 = norm2 + l2(param, weights_SP[n])
		# loss = loss_func(Z_hat, targets) + 0.5 * 0.5 * norm2  # (1/2 * alpha * l2(weights))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		outputs = outputs.squeeze()
		if step > 16261:
			sheet = file.add_sheet(str(step), cell_overwrite_ok=True)
			for j in range(len(outputs)):
				sheet.write(j, 0, outputs[j].item())
	# sheet1 = file.add_sheet('loss')
	# for j in range(len(his_loss)):
	# 	sheet1.write(j, 0, his_loss[j])
	ffilename = 'dnn-test.xls'
	file.save(ffilename)




def outlierfactor():
	print('Outlier Factor...')
	estimator.load_state_dict(torch.load('final_params.pkl'))
	EZ = torch.zeros(sys_data1.shape[0], sys_data1.shape[1])
	states = torch.zeros(sys_data1.shape[0], 235)
	for step, (batch_train, batch_targets) in enumerate(loader1):
		batch_train = batch_train.view(1, batch_train.shape[0], batch_train.shape[1])
		inputs = Variable(batch_train, requires_grad=True).to(device)
		targets = Variable(batch_targets).to(device)
		
		outputs = estimator(inputs)
		Z_hat = power_flow(outputs)

		EZ[step, :] = Z_hat
		states[step, :] = outputs
	diff = (EZ-sys_data1).mul(EZ-sys_data1)

	dfEZ = pd.DataFrame(EZ.detach().numpy())
	dfdiff= pd.DataFrame(diff.detach().numpy())
	dfstates = pd.DataFrame(states.detach().numpy())
	writer = ExcelWriter('TrainingData_Excel.xlsx')
	dfEZ.to_excel(writer, 'Zhat', index=False)
	dfdiff.to_excel(writer, 'Difference', index=False)
	dfstates.to_excel(writer, 'States', index=False)
	writer.save()

	# OF = torch.sum(diff,0) / sys_data1.shape[0]
	# file = xlwt.Workbook()
	# sheet = file.add_sheet('OutlierFactor')
	# for j in range(len(OF)):
	# 	sheet.write(j, 0, OF[j].item())
	# ffilename = 'TrainOutlierFactor.xls'
	# file.save(ffilename)


# def outlierfactor():
# 	print('Outlier Factor...')
# 	estimator.load_state_dict(torch.load('final_params.pkl'))
# 	EZ = torch.zeros(stream_data.shape[0], stream_data.shape[1])
# 	states = torch.zeros(stream_data.shape[0], 235)
# 	for step, (batch_train, batch_targets) in enumerate(stream_loader):
# 		batch_train = batch_train.view(1, batch_train.shape[0], batch_train.shape[1])
# 		inputs = Variable(batch_train, requires_grad=True).to(device)
# 		targets = Variable(batch_targets).to(device)
#
# 		outputs = estimator(inputs)
# 		Z_hat = power_flow(outputs)
#
# 		EZ[step, :] = Z_hat
# 		states[step, :] = outputs
#
# 	diff = (EZ - stream_data).mul(EZ - stream_data)
#
# 	dfEZ = pd.DataFrame(EZ.detach().numpy())
# 	dfdiff= pd.DataFrame(diff.detach().numpy())
# 	dfstates = pd.DataFrame(states.detach().numpy())
# 	writer = ExcelWriter('TestingData_Excel.xlsx')
# 	dfEZ.to_excel(writer, 'Zhat', index=False)
# 	dfdiff.to_excel(writer, 'Difference', index=False)
# 	dfstates.to_excel(writer, 'States', index=False)
# 	writer.save()
#
#
#
#
# 	# OF = torch.sum(diff, 0) / stream_data.shape[0]
# 	# file = xlwt.Workbook()
# 	# sheet = file.add_sheet('OutlierFactorTest')
# 	# for j in range(len(OF)):
# 	# 	sheet.write(j, 0, OF[j].item())
# 	# ffilename = 'TestOutlierFactor.xls'
# 	# file.save(ffilename)





def main():
	# batch_train_params()
	train()
	# pfm_test()
	# outlierfactor()


if __name__ == "__main__":
	main()
