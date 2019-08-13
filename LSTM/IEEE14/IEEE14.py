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


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(torch.cuda.get_device_name(torch.cuda.current_device()))

xls = pd.ExcelFile('measurements.xlsx')
sys_data = pd.read_excel(xls, sheet_name='Data', header=None)
sys_data = sys_data.as_matrix()

G = pd.read_excel(xls, sheet_name='Real', header=None)
G = G.as_matrix()
# Imag part
imag = pd.read_excel(xls, sheet_name='Imag', header=None)
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
nBus = 14  # Bus # of targeted system
M = sys_data.shape[1]  # Measurements #
nSet = sys_data.shape[0]  # Sets # of sys_data
N = 2 * nBus - 1  # State variables #
ref_ang = 0 * math.pi / 180  # reference bus angle

# Use Variable container
G = Variable(torch.from_numpy(G).float()).to(device)
B = Variable(torch.from_numpy(B).float()).to(device)

# Mini-batch DataLoader
# Hyper parameters
BATCH_SIZE = 32
LR = 1e-4
EPOCH = 100

sys_data1 = torch.from_numpy(sys_data[5:]).float()
iter_size = len(sys_data1) //BATCH_SIZE
torch_dataset1 = Data.TensorDataset(sys_data1, sys_data1)
loader1 = Data.DataLoader(dataset=torch_dataset1,  # torch TensorDataset format
                         batch_size=BATCH_SIZE,  # mini batch size
                         shuffle=False,  # random shuffle for training
                         num_workers=2,  # sub-processes for loading data
                         drop_last=True,
                         )


sys_data2 = torch.from_numpy(sys_data[5:869]).float()

true_data = pd.read_excel(xls, sheetname='WLS_Train', header=None)
true_data = true_data.as_matrix()
true_data = torch.from_numpy(true_data[:864]).float()

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
		self.lstm = nn.LSTM(input_size=M,
		                    hidden_size=128,
		                    num_layers=2,
		                    batch_first=False,
		                    )
		self.out = nn.Linear(128, N)
	
	def forward(self, inputs, hidden):
		r_out, hidden = self.lstm(inputs, hidden)
		outputs = self.out(r_out[-1, :, :])
		return outputs, hidden


def repackage_hidden(h):
	"Wraps hidden states in new Tensors, to detach them from their history."
	if h is None:
		return h
	return [state.detach() for state in h]


def weight_init(m):
	'''
	Usage:
		model = Model()
		model.apply(weight_init)
	'''
	if isinstance(m, nn.Linear):#######################################
		init.xavier_normal_(m.weight.data)
		init.normal_(m.bias.data)
	elif isinstance(m, nn.LSTM): #####################################
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)



# Reload the module after initialization
estimator = Net()
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
	outputs = outputs.permute(1, 0)
	Z_hat = Variable(torch.zeros(M, 1)).to(device)
	A = Variable(ref_ang * torch.ones(nBus, 1)).to(device)
	A[1:nBus] = outputs[0:nBus - 1]  # Reference Bus #1
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
	outputs = outputs.unsqueeze(2)
	Z_hat = Variable(torch.zeros(BATCH_SIZE, M, 1)).to(device)
	A = Variable(ref_ang * torch.ones(BATCH_SIZE, nBus, 1)).to(device)
	A[:, 1:nBus] = outputs[:, 0:nBus - 1]  # voltage angle
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


# def batch_train_params():
# 	# initialize the parameters of NETWORK
# 	hidden = None
#
# 	with trange(0, 300) as numEpoch:
# 		for epoch in numEpoch:
# 			for step, (batch_train, batch_target) in enumerate(loader):
# 				batch_train = batch_train.view(1, batch_train.shape[0], batch_train.shape[1])
# 				inputs = Variable(batch_train, requires_grad=True).to(device)
# 				targets = Variable(batch_target).to(device)
# 				hidden = repackage_hidden(hidden)
# 				outputs, hidden = estimator(inputs, hidden)
# 				loss = loss_func(outputs, targets)
# 				optimizer.zero_grad()
# 				loss.backward()
# 				optimizer.step()
# 			numEpoch.set_postfix(Loss='%0.32f' % loss)
# 	torch.save(estimator.state_dict(), 'initial_batch_params_TS.pkl')

def batch_train_params():
	print('Start Initializing...')
	estimator.apply(weight_init)
	hidden = None
	with trange(0, 500) as numEpoch:
		for epoch in numEpoch:
			for step, (batch_train, batch_target) in enumerate(loader2):
				batch_train = batch_train.view(1, batch_train.shape[0], batch_train.shape[1])
				inputs = Variable(batch_train, requires_grad=True).to(device)
				targets = Variable(batch_target).to(device)
				
				hidden = repackage_hidden(hidden)
				optimizer.zero_grad()
				outputs, hidden = estimator(inputs, hidden)
				loss = loss_func(outputs, targets)
				loss.backward()
				optimizer.step()
			numEpoch.set_postfix(Loss='%0.8f' % loss)
	torch.save(estimator.state_dict(), 'initial_batch_params.pkl')
	[h,c] = hidden
	file = xlwt.Workbook()
	for i in range(h.shape[0]):
		sheetname = 'h-' + str(i)
		sheet = file.add_sheet(sheetname, cell_overwrite_ok=True)
		for j in range(h.shape[1]):
			for r in range(h.shape[2]):
				sheet.write(j, r, h[i,j,r].item())
	
	for i in range(c.shape[0]):
		sheetname = 'c-' + str(i)
		sheet1 = file.add_sheet(sheetname, cell_overwrite_ok=True)
		for j in range(c.shape[1]):
			for r in range(c.shape[2]):
				sheet1.write(j, r, c[i,j,r].item())
	
	ffilename = 'hidden.xls'
	file.save(ffilename)
	
	return hidden

def train():
	print('Start Training...')
	
	h_1 = pd.read_excel('hidden.xls', sheet_name='h-0', header=None)
	h_2 = pd.read_excel('hidden.xls', sheet_name='h-1', header=None)
	c_1 = pd.read_excel('hidden.xls', sheet_name='c-0', header=None)
	c_2 = pd.read_excel('hidden.xls', sheet_name='c-1', header=None)
	
	h_1 = h_1.as_matrix()
	h_1 = torch.from_numpy(h_1).float().to(device)
	h_1 = torch.unsqueeze(h_1,0)
	h_2 = h_2.as_matrix()
	h_2 = torch.from_numpy(h_2).float().to(device)
	h_2 = torch.unsqueeze(h_2, 0)
	c_1 = c_1.as_matrix()
	c_1 = torch.from_numpy(c_1).float().to(device)
	c_1 = torch.unsqueeze(c_1, 0)
	c_2 = c_2.as_matrix()
	c_2 = torch.from_numpy(c_2).float().to(device)
	c_2 = torch.unsqueeze(c_2, 0)
	
	h = torch.cat((h_1,h_2),dim=0)
	c = torch.cat((c_1, c_2), dim=0)
	
	hidden =[h,c]
	
	estimator.load_state_dict(torch.load('initial_batch_params.pkl'))
	scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4, step_size=300, mode='triangular2')
	
	with trange(0, EPOCH) as numEPoch:
		for epoch in numEPoch:
			total_loss = 0
			scheduler.step()
			optimizer.zero_grad()
			for step, (batch_train, batch_targets) in enumerate(loader1):
				batch_train = batch_train.view(1, batch_train.shape[0], batch_train.shape[1])
				inputs = Variable(batch_train, requires_grad=True).to(device)
				targets = Variable(batch_targets).to(device)
				
				hidden = repackage_hidden(hidden)
				
				optimizer.zero_grad()
				outputs, hidden = estimator(inputs, hidden)
				Z_hat = power_flow_batch(outputs)
				loss = loss_func(Z_hat, targets)/iter_size
				total_loss += loss
				loss.backward()
			optimizer.step()
			numEPoch.set_postfix(Loss='%0.8f' % total_loss)  # print error
			# print('Loss=%0.8f' % loss)
	torch.save(estimator.state_dict(), 'final_params.pkl')
	
	[h, c] = hidden
	file = xlwt.Workbook()
	for i in range(h.shape[0]):
		sheetname = 'h-' + str(i)
		sheet = file.add_sheet(sheetname, cell_overwrite_ok=True)
		for j in range(h.shape[1]):
			for r in range(h.shape[2]):
				sheet.write(j, r, h[i, j, r].item())
	
	for i in range(c.shape[0]):
		sheetname = 'c-' + str(i)
		sheet1 = file.add_sheet(sheetname, cell_overwrite_ok=True)
		for j in range(c.shape[1]):
			for r in range(c.shape[2]):
				sheet1.write(j, r, c[i, j, r].item())
	
	ffilename = 'hiddentrain.xls'
	file.save(ffilename)
	
	return hidden


# Load streaming data
stream_data = pd.read_excel(xls, sheet_name='TestData', header=None)
stream_data = stream_data.as_matrix()
stream_data = torch.from_numpy(stream_data).float()
stream_torch_dataset = Data.TensorDataset(stream_data, stream_data)
stream_loader = Data.DataLoader(dataset=stream_torch_dataset, batch_size=1, shuffle=False,
                                num_workers=2,  # sub-processes for loading data
                                )


def pfm_test():
	estimator.load_state_dict(torch.load('final_params.pkl'))
	
	h_1 = pd.read_excel('hiddentrain.xls', sheet_name='h-0', header=None)
	h_2 = pd.read_excel('hiddentrain.xls', sheet_name='h-1', header=None)
	c_1 = pd.read_excel('hiddentrain.xls', sheet_name='c-0', header=None)
	c_2 = pd.read_excel('hiddentrain.xls', sheet_name='c-1', header=None)
	
	h_1 = h_1.as_matrix()
	h_1 = torch.from_numpy(h_1).float().to(device)
	h_1 = torch.unsqueeze(h_1, 0)
	h_2 = h_2.as_matrix()
	h_2 = torch.from_numpy(h_2).float().to(device)
	h_2 = torch.unsqueeze(h_2, 0)
	c_1 = c_1.as_matrix()
	c_1 = torch.from_numpy(c_1).float().to(device)
	c_1 = torch.unsqueeze(c_1, 0)
	c_2 = c_2.as_matrix()
	c_2 = torch.from_numpy(c_2).float().to(device)
	c_2 = torch.unsqueeze(c_2, 0)
	
	h = torch.cat((h_1[:,-1,:].unsqueeze(1).contiguous(), h_2[:,-1,:].unsqueeze(1).contiguous()), dim=0)
	c = torch.cat((c_1[:,-1,:].unsqueeze(1).contiguous(), c_2[:,-1,:].unsqueeze(1).contiguous()), dim=0)
	
	hidden = [h, c]
	
	for param_group in optimizer.param_groups:
		param_group['lr'] = 1e-8
	print('Online Training...')
	file = xlwt.Workbook()

	for step, (batch_train, batch_targets) in enumerate(stream_loader):
		batch_train = batch_train.view(1, batch_train.shape[0], batch_train.shape[1])
		inputs = Variable(batch_train, requires_grad=True).to(device)
		targets = Variable(batch_targets).to(device)

		hidden = repackage_hidden(hidden)

		outputs, hidden = estimator(inputs, hidden)
		Z_hat = power_flow(outputs)           ###power_flow() is much faster than power_flow_batch()
		loss = loss_func(Z_hat, targets)
		if loss.item() < 5e-4:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		outputs = outputs.squeeze()
		if step > 16261:
			sheet = file.add_sheet(str(step), cell_overwrite_ok=True)
			for j in range(len(outputs)):
				sheet.write(j, 0, outputs[j].item())

	ffilename = 'dnn-test.xls'
	file.save(ffilename)
	

def main():
	# hidden = batch_train_params()
	# hidden = train()
	# hidden = (hidden[0][:, -1, :].unsqueeze(1).contiguous(), hidden[1][:, -1, :].unsqueeze(1).contiguous())
	pfm_test()


if __name__ == "__main__":
	main()
