import numpy as np
import reader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pandas as pd
import glob
import copy


def getMean(X):
	return np.mean(X,1).reshape(-1,1)

class Network():
	def __init__(self, setup, eta=0.1, regTerm = 0.1, seqLength = 25):
		self.m = setup[1]
		sig = 0.01
		self.W = np.array([np.random.random()*0.01 for k in range(self.m*self.m)], dtype=np.float64).reshape(self.m,self.m)
		self.U = np.array([np.random.random()*0.01 for k in range(self.m*setup[0])],dtype=np.float64).reshape(self.m, setup[0])							
		self.V = np.array([np.random.random()*0.01 for k in range(self.m*setup[0])],dtype=np.float64).reshape(setup[0],self.m)							
		self.b = np.array([0.0 for k in range(self.m)], dtype=np.float64).reshape(self.m,1)
		self.c = np.array([0.0 for k in range(setup[0])], dtype=np.float64).reshape(setup[0],1)
		self.seqLength = seqLength
		self.dh = reader.DataHandler()
		self.eta = eta
		self.regTerm = regTerm
		self.e = 1e-5

		#adagrad stuff
		self.mW = np.array([0.0 for k in range(self.m*self.m)], dtype=np.float64).reshape(self.m,self.m)
		self.mU = np.array([0.0 for k in range(self.m*setup[0])],dtype=np.float64).reshape(self.m, setup[0])
		self.mV = np.array([0.0 for k in range(self.m*setup[0])],dtype=np.float64).reshape(setup[0],self.m)
		self.mb = np.array([0.0 for k in range(self.m)], dtype=np.float64).reshape(self.m,1)
		self.mc = np.array([0.0 for k in range(setup[0])], dtype=np.float64).reshape(setup[0],1)

	def forward(self, X):
		a = []
		h = []
		o = []
		p = []
		h = []
		for t in range(X.shape[0]):
			xt = X[t].reshape(-1,1)
			apart = np.dot(self.U,xt) + self.b
			if t > 0:
				apart += np.dot(self.W,h[t-1])
			a.append(apart)
			h.append(self.tanh(a[-1]))
			o.append(np.dot(self.V,h[-1]) + self.c)
			p.append(self.softMax(o[-1]))
		return np.asarray(a).reshape(-1,self.W.shape[0]), np.asarray(h).reshape(-1,self.W.shape[0]), np.asarray(o).reshape(-1,dh.len), np.asarray(p).reshape(-1,dh.len) #Fix this at some point

	def drawNewLetter(self, p):
		cp = np.cumsum(p)
		a = np.random.random()
		ixs = np.argwhere(cp - a > 0)
		if len(ixs) == 0:
			index = len(p)
		else:
			index = ixs[0].item()
		return self.dh.getOneHot(index), self.dh.getLetter(index)


	def tanh(self, X):
		nexp = np.exp(X)
		mexp = np.exp(-X)
		return (nexp - mexp)/(nexp + mexp)

	def deltatanh(self, X):
		return 1 - np.power(X,2)

	def softMax(self,S2):
		P = np.zeros((S2.shape[0],S2.shape[1]))
		for c in range(len(S2.T)):
			maxx = np.max(S2[:,c])
			row = S2[:,c] - maxx
			suma = np.sum(np.exp(row))
			P[:,c] = np.exp(row)/suma
		return P

	def getL(self, P, Y):
		total = 0.0
		for i in range(len(P)):
			total += - np.log(np.dot(Y[i],P[i])).item()
		return total

	def generateString(self, startSeq, length=100):
		x = copy.copy(startSeq)
		encodedString = x
		for i in range(length):
			_,_,_,p = self.forward(x)
			onehot, letter = self.drawNewLetter(p)
			x = x[1:]
			encodedString = np.concatenate((encodedString,onehot),axis=0)
			x = np.concatenate((x,onehot), axis=0)
		string = ""
		for en in encodedString:
			string += dh.encodedToLetter(en)
		return string

	def calculateLoss(self,X,Y):
		a,h,o,p = self.forward(X)
		loss = self.getL(p,Y)
		return loss


	def calculateGradient(self,x,y):
		dldw = np.zeros(self.W.shape)
		dldb = np.zeros(self.b.shape)
		dldu = np.zeros(self.U.shape)
		dldv = np.zeros(self.V.shape)
		dldc = np.zeros(self.c.shape)
		dlda = np.zeros((1,self.W.shape[0]))
		a,h,o,p = self.forward(x)
		for t in reversed(range(x.shape[0])):
			dldo = -(y[t] - p[t]).T
			dldc += dldo.reshape(-1,1)
			dldv += np.outer(dldo,h[t])
			dldh = np.dot(dldo,self.V) + np.dot(dlda,self.W)
			dlda = np.dot(dldh, np.diagflat(self.deltatanh(a[t])))
			dldw += np.outer(dlda.T, h[t-1])
			dldu += np.outer(dlda.T,x[t])
			dldb += dlda.T

		return dldw, dldb, dldu, dldv, dldc

	def updateWithBatch(self,x,y):
		dldw = np.zeros(self.W.shape)
		dldb = np.zeros(self.b.shape)
		dldu = np.zeros(self.U.shape)
		dldv = np.zeros(self.V.shape)
		dldc = np.zeros(self.c.shape)
		dldw, dldb, dldu, dldv, dldc = network.calculateGradient(x,y)
		
		self.mW = self.mW + np.power(dldw,2)
		self.mb = self.mb + np.power(dldb,2)
		self.mU = self.mU + np.power(dldu,2)
		self.mV = self.mV + np.power(dldv,2)
		self.mc = self.mc + np.power(dldc,2)
		
		self.W = self.updateParameters(self.mW,self.W)
		self.b = self.updateParameters(self.mb,self.b)
		self.U = self.updateParameters(self.mU,self.U)
		self.V = self.updateParameters(self.mV,self.V)
		self.c = self.updateParameters(self.mc,self.c)
		

	def updateParameters(self, m, g):
		return g - (self.eta/ np.sqrt(m+self.e))*g




dh = reader.DataHandler()

network = Network([dh.len,100,dh.len])
x,y = dh.getInputOutput(0,25)

for i in range(1000):
	network.updateWithBatch(x,y)
	if i % 10 == 0:
		print(network.calculateLoss(x,y))

network.generateString(x[1:2])


