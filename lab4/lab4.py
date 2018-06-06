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
	def __init__(self, setup, eta=0.1, seqLength = 25):
		self.m = setup[1]
		sig = 0.01
		self.W = np.random.randn(self.m,self.m) * sig
		self.U = np.random.randn(self.m, setup[0]) * sig				
		self.V = np.random.randn(setup[0],self.m) * sig	
		self.b = np.zeros((self.m,1))
		self.c = np.zeros((setup[0],1))

		self.seqLength = seqLength
		self.dh = reader.DataHandler()
		self.eta = eta
		self.e = 1e-8
		self.prevh = np.zeros((self.W.shape[0],1))
		#adagrad stuff
		self.mW = np.zeros((self.m,self.m))
		self.mU = np.zeros((self.m, setup[0]))
		self.mV = np.zeros((setup[0],self.m))
		self.mb = np.zeros((self.m,1))
		self.mc = np.zeros((setup[0],1))

	def forward(self, X):
		p = []
		h = [self.prevh]
		for t in range(X.shape[1]):
			apart = np.dot(self.U,X[:,t:t+1]) + np.dot(self.W,h[-1]) + self.b
			h.append(np.tanh(apart))
			o = np.dot(self.V,h[-1]) + self.c
			p.append(self.softMax(o))
		self.prevh = h[-1]
		return h, p #Fix this at some point


	def resetState(self):
		self.prevh = np.zeros(self.prevh.shape)

	def drawNewLetter(self, p):
		cp = np.cumsum(p)
		a = np.random.random()
		ixs = np.argwhere(cp - a > 0)
		if len(ixs) == 0:
			index = len(p)-1
		else:
			index = ixs[0].item()
		return self.dh.getOneHot(index).reshape(-1,1), self.dh.getLetter(index)


	def tanh(self, X):
		nexp = np.exp(X)
		mexp = np.exp(-X)
		return (nexp - mexp)/(nexp + mexp)

	def deltatanh(self, X):
		return 1 - np.power(X,2)

	def softMax(self,x):
		e_x = np.exp(x - np.max(x))
		return e_x/e_x.sum()
		

	def getL(self, P, Y):
		total = 0.0

		for i in range(Y.shape[1]):
			total -=  np.log(np.dot(Y[:,i:i+1].T, P[i])).item()
		return total

	def generateString(self, length=100):
		trainh = copy.copy(self.prevh)
		string = ""
		xi = np.random.randint(0,80)
		x = np.zeros((self.dh.len,1))
		x[xi] = 1.
		for i in range(length):
			_,p = self.forward(x)
			onehot, letter = self.drawNewLetter(p)
			string += letter
			x = onehot
	
		return string

	def calculateLoss(self,X,Y):
		h,p = self.forward(X)
		#print(p)
		loss = self.getL(p,Y)
		return loss


	def calculateGradient(self,x,y):
		dldw = np.zeros(self.W.shape)
		dldb = np.zeros(self.b.shape)
		dldu = np.zeros(self.U.shape)
		dldv = np.zeros(self.V.shape)
		dldc = np.zeros(self.c.shape)
		h,p = self.forward(x)
		dldh = np.zeros(h[0].shape)

		for t in reversed(range(x.shape[1])):
			dldo = p[t]-y[:,t:t+1]
			dldc += dldo
			dldv += np.dot(dldo,h[t+1].T)
			dldh = (self.deltatanh(h[t+1])) * (np.dot(self.V.T, dldo) + dldh)

			dldw += np.dot(dldh, h[t].T)
			dldu += np.dot(dldh, x[:,t:t+1].T)
			dldb += dldh

			dldh = np.dot(self.W.T, dldh)

		

		dldw = np.clip(dldw,-5.,5.)
		dldb = np.clip(dldb,-5.,5.)
		dldu = np.clip(dldu,-5.,5.)
		dldv = np.clip(dldv,-5.,5.)
		dldc = np.clip(dldc,-5.,5.)

		return dldw, dldb, dldu, dldv, dldc

	def updateWithBatch(self,x,y):
		dldw, dldb, dldu, dldv, dldc = network.calculateGradient(x,y)
		#print(np.max(dldw))
		#print(np.max(dldb))
		#print(np.max(dldu))
		#print(np.max(dldv))
		#print(np.max(dldc))
		#print("------")

		self.mW +=  np.power(dldw,2)
		self.mb +=  np.power(dldb,2)
		self.mU +=  np.power(dldu,2)
		self.mV +=  np.power(dldv,2)
		self.mc +=  np.power(dldc,2)
		
		self.W += self.updateParameters(self.mW,self.W)
		self.b += self.updateParameters(self.mb,self.b)
		self.U += self.updateParameters(self.mU,self.U)
		self.V += self.updateParameters(self.mV,self.V)
		self.c += self.updateParameters(self.mc,self.c)
		

	def updateParameters(self, m, g):
		return -self.eta*g/np.sqrt(m+self.e)




if __name__ == "__main__":
	dh = reader.DataHandler()
	print("data aquired")


	network = Network([dh.len,100,dh.len])
	
	counter = 0
	i = 0
	
	while counter < 2:
		x,y = dh.getInputOutput(i,1)
		network.updateWithBatch(x,y)
		if i % 1 == 0:
			print("Loss: " + str(network.calculateLoss(x,y)))
		counter += 1
		i += network.seqLength
	
	x,y = dh.getInputOutput(0,25)
	print(network.generateString())
	


