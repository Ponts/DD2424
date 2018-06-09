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
	def __init__(self, setup, eta=0.1, seqLength = 25, mode="Potter"):
		if mode == "Potter":
			self.dh = reader.DataHandler()
		else:
			self.dh = reader.DataHandler(mode="Trump")
		self.m = setup
		sig = 0.01
		self.W = np.random.rand(self.m,self.m) * sig
		self.U = np.random.rand(self.m, self.dh.len) * sig				
		self.V = np.random.rand(self.dh.len,self.m) * sig	
		self.b = np.zeros((self.m,1))
		self.c = np.zeros((self.dh.len,1))

		self.seqLength = seqLength
		
		self.eta = eta
		self.e = 1e-08
		self.prevh = np.zeros((self.W.shape[0],1))

		#adagrad stuff
		self.mW = np.zeros(self.W.shape)
		self.mU = np.zeros(self.U.shape)
		self.mV = np.zeros(self.V.shape)
		self.mb = np.zeros(self.b.shape)
		self.mc = np.zeros(self.c.shape)




	def forward(self, X,Y):
		p = []
		h = [self.prevh]
		for t in range(X.shape[1]):
			apart = np.dot(self.U,X[:,t:t+1]) + np.dot(self.W,h[-1]) + self.b
			hidden = np.tanh(apart)
			h.append(hidden)
			o = np.dot(self.V,hidden) + self.c
			p.append(self.softMax(o))

		self.prevh = h[-1]

		loss = self.getL(p,Y)

		return h, p, loss


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
		return e_x/e_x.sum(axis=0)
		

	def getL(self, P, Y):
		total = 0.0
		for i in range(Y.shape[1]):
			total -=  np.log(np.dot(Y[:,i:i+1].T, P[i])).item()
		return total

	def generateString(self, length=100):
		trainh = copy.copy(self.prevh)
		string = ""
		xi = np.random.randint(0,self.dh.len)
		x = np.zeros((self.dh.len,1))
		x[xi] = 1.
		for i in range(length):
			_,p,_ = self.forward(x,x) #FIX
			onehot, letter = self.drawNewLetter(p)
			string += letter
			x = onehot
		self.prevh = copy.copy(trainh)
	
		return string

	def generateTrump(self, length=140):
		string = ""
		xi = np.random.randint(0,self.dh.len)
		x = np.zeros((self.dh.len,1))
		x[xi] = 1.
		for i in range(length):
			_,p,_ = self.forward(x,x) #FIX
			onehot, letter = self.drawNewLetter(p)
			if letter == self.dh.EOT:
				self.resetState()
				return string
			string += letter
			x = onehot
		self.resetState()
		return string

	def calculateLoss(self,X,Y):
		h,p,loss = self.forward(X,Y)
		return loss




	def calculateGradient(self,x,y):
		
		dldw = np.zeros(self.W.shape)
		dldb = np.zeros(self.b.shape)
		dldu = np.zeros(self.U.shape)
		dldv = np.zeros(self.V.shape)
		dldc = np.zeros(self.c.shape)
		h,p,loss = self.forward(x,y)
		dlda = np.zeros(h[-1].shape)
		

		for t in reversed(range(x.shape[1])):
			dldo = p[t] - y[:,t:t+1]
			dldv += np.matmul(dldo, h[t+1].T)
			dldc += dldo

			dldh = np.matmul(self.V.T, dldo) + np.matmul(self.W.T, dlda)
			dlda = np.multiply(dldh, self.deltatanh(h[t+1]))

			dldw += np.matmul(dlda, h[t].T)
			dldb += dlda
			dldu += np.matmul(dlda, x[:,t:t+1].T)

		dldw = np.clip(dldw, -5, 5)
		dldb = np.clip(dldb, -5, 5)
		dldu = np.clip(dldu, -5, 5)
		dldv = np.clip(dldv, -5, 5)
		dldc = np.clip(dldc, -5, 5)
		

		return dldw, dldb, dldu, dldv, dldc, loss

	def updateWithBatch(self,x,y):
		dldw, dldb, dldu, dldv, dldc, loss = network.calculateGradient(x,y)


		self.mW +=  np.power(dldw,2)
		self.mb +=  np.power(dldb,2)
		self.mU +=  np.power(dldu,2)
		self.mV +=  np.power(dldv,2)
		self.mc +=  np.power(dldc,2)
		
		self.W -= self.eta * np.divide(dldw, np.sqrt(self.mW + self.e))
		self.b -= self.eta * np.divide(dldb, np.sqrt(self.mb + self.e))
		self.U -= self.eta * np.divide(dldu, np.sqrt(self.mU + self.e))
		self.V -= self.eta * np.divide(dldv, np.sqrt(self.mV + self.e))
		self.c -= self.eta * np.divide(dldc, np.sqrt(self.mc + self.e))
		return loss


	def trainPotter(self):
		epochLim = 100
		counter = 0
		index = self.seqLength
		iterN = 1
		x,y = self.dh.getInputOutput(0,self.seqLength)
		smoothLoss = self.calculateLoss(x,y)
		self.updateWithBatch(x,y)
		print("Loss: " + str(smoothLoss))

		while counter < epochLim:
			x,y = self.dh.getInputOutput(index, self.seqLength)
			loss = self.updateWithBatch(x,y)
			smoothLoss = smoothLoss * 0.999 + loss * 0.001
			if iterN % 100 == 0:
				print("Iteration: " + str(iterN) + ", Loss: " + str(smoothLoss))
			if iterN % 500 == 0:
				print(self.generateString())
			if index > len(self.dh.data) - self.seqLength*2:
				index = 0
				counter += 1
				self.resetState()
				print("EPOCH DONE - STARTING OVER")
			else:
				iterN += 1
				index += self.seqLength
		print(network.generateString(200))



	def trainTrump(self):
		epochLim = 100
		counter = 0
		index = self.seqLength
		iterN = 0
		x,y = self.dh.getInputOutput(0,self.seqLength)
		eotI = np.argwhere(x[self.dh.endChar] == 1)
		if  eotI != -1: #End of tweet char in text
			x = x[:eotI[0]]
			y = y[:eotI[0]]
			index = eotI[0]
		smoothLoss = self.calculateLoss(x,y)
		self.updateWithBatch(x,y)
		print("Loss: " + str(smoothLoss))

		while counter < epochLim:
			self.resetState()
			x,y = self.dh.getInputOutput(index, self.seqLength)
			eotI = np.argwhere(x[self.dh.endChar] == 1)
			if len(eotI): #End of tweet char in text
				x = x[:,:eotI.item(0)]
				y = y[:,:eotI.item(0)]
			loss = self.updateWithBatch(x,y)
			smoothLoss = smoothLoss * 0.999 + loss * 0.001
			if iterN % 100 == 0:
				print("Iteration: " + str(iterN) + ", Loss: " + str(smoothLoss))
			if iterN % 500 == 0:
				print(self.generateTrump())
			if index > len(self.dh.data) - self.seqLength*2 - 1:
				index = 0
				counter += 1
				self.resetState()
				print("EPOCH DONE - STARTING OVER")
			else:
				iterN += 1
				if len(eotI):
					index += eotI.item(0)+1
				else:
					index += self.seqLength
		print(network.generateTrump())
		


if __name__ == "__main__":
	#Here we train for Trump
	network = Network(100, mode="Trump")
	network.trainTrump()

	#Here we train for Potter
	#network = Network(100)
	#network.trainPotter()

