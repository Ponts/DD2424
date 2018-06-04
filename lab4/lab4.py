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
		self.W = np.array([np.random.random()*sig for k in range(self.m*self.m)]).reshape(self.m,self.m)
		self.U = np.array([np.random.random()*sig for k in range(self.m*setup[0])]).reshape(self.m, setup[0])							
		self.V = np.array([np.random.random()*sig for k in range(self.m*setup[0])]).reshape(setup[0],self.m)							
		self.b = np.array([0.0 for k in range(self.m)]).reshape(self.m,1)
		self.c = np.array([0.0 for k in range(setup[0])]).reshape(setup[0],1)
		self.seqLength = seqLength
		self.dh = reader.DataHandler()
		self.eta = eta
		self.e = 1e-8
		self.prevh = np.zeros((self.W.shape[0],1))
		#adagrad stuff
		self.mW = np.array([0.0 for k in range(self.m*self.m)]).reshape(self.m,self.m)
		self.mU = np.array([0.0 for k in range(self.m*setup[0])]).reshape(self.m, setup[0])
		self.mV = np.array([0.0 for k in range(self.m*setup[0])]).reshape(setup[0],self.m)
		self.mb = np.array([0.0 for k in range(self.m)]).reshape(self.m,1)
		self.mc = np.array([0.0 for k in range(setup[0])]).reshape(setup[0],1)

	def forward(self, X):
		h = []
		p = []
		h = [self.prevh]
		for t in range(X.shape[0]):
			xt = X[t].reshape(-1,1)
			apart = np.dot(self.U,xt) + np.dot(self.W,h[-1]) + self.b
			h.append(np.tanh(apart))
			o = np.dot(self.V,h[-1]) + self.c
			p.append(self.softMax(o))
		self.prevh = h[-1]
		return np.asarray(h).reshape(-1,self.W.shape[0]), np.asarray(p).reshape(-1,self.dh.len) #Fix this at some point

	def resetState(self):
		self.prevh = np.zeros((self.W.shape[0],1))

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
		exp = np.exp(S2)
		ret = exp/np.sum(exp)
		return ret

	def getL(self, P, Y):
		total = 0.0
		for i in range(len(P)):
			total += - np.log(np.dot(Y[i],P[i])).item()
		return total

	def generateString(self, startSeq, length=100):
		x = copy.copy(startSeq)
		encodedString = x
		for i in range(length):
			_,p = self.forward(x)
			onehot, letter = self.drawNewLetter(p)
			x = x[1:]
			encodedString = np.concatenate((encodedString,onehot),axis=0)
			x = np.concatenate((x,onehot), axis=0)
		string = ""
		for en in encodedString:
			string += dh.encodedToLetter(en)
		return string

	def calculateLoss(self,X,Y):
		h,p = self.forward(X)
		loss = self.getL(p,Y)
		return loss


	def calculateGradient(self,x,y):
		dldw = np.zeros(self.W.shape)
		dldb = np.zeros(self.b.shape)
		dldu = np.zeros(self.U.shape)
		#dldv = np.zeros(self.V.shape)
		#dldc = np.zeros(self.c.shape)
		dlda = np.zeros((1,self.W.shape[0]))
		h,p = self.forward(x)

		dldo = p - y
		dldc = np.sum(dldo.T,axis=1).reshape(-1,1)
		dldv = np.dot(dldo.T, h[1:])
		
		for t in reversed(range(x.shape[0])):
			#dldo = (p[t]-y[t]).reshape(-1,1)
			#dldc += dldo
			#dldv += np.dot(dldo,h[t+1:t+2])
			dldh = np.dot(dldo[t:t+1], self.V) + np.dot(dlda, self.W.T)
			dlda = np.multiply(self.deltatanh(h[t+1:t+2]), dldh)

			dldw += np.dot(dlda.T, h[t:t+1])
			dldu += np.dot(dlda.T, x[t:t+1])

			dldb += dlda.T
			
		#dldw = np.clip(dldw,-1.,1.)
		#dldb = np.clip(dldb,-1.,1.)
		#dldu = np.clip(dldu,-1.,1.)
		#dldv = np.clip(dldv,-1.,1.)
		#dldc = np.clip(dldc,-1.,1.)

		return dldw, dldb, dldu, dldv, dldc

	def updateWithBatch(self,x,y):
		dldw, dldb, dldu, dldv, dldc = network.calculateGradient(x,y)
		#print(np.max(dldw))
		#print(np.max(dldb))
		#print(np.max(dldu))
		#print(np.max(dldv))
		#print(np.max(dldc))
		#print("------")

		self.mW = self.mW + np.power(dldw,2)
		self.mb = self.mb + np.power(dldb,2)
		self.mU = self.mU + np.power(dldu,2)
		self.mV = self.mV + np.power(dldv,2)
		self.mc = self.mc + np.power(dldc,2)
		
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
		x,y = dh.getInputOutput(i,network.seqLength)
		network.updateWithBatch(x,y)

		if i % 1 == 0:
			print("Loss: " + str(network.calculateLoss(x,y)))
		counter += 1
		i += network.seqLength

	x,y = dh.getInputOutput(0,25)
	print(network.generateString(x[0:25]))
	


