import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pandas as pd
import glob
filepath = "cifar-10-batches-py/"

def unpickle(file):
	import pickle
	with open(filepath + file, 'rb') as fo:
		dict = pickle.load(fo, encoding='latin1')
	return dict

def getData(file):
	pickle = unpickle(file)
	filenames = unpickle("batches.meta")
	X = pickle.get("data").T
	label = np.array(pickle.get("labels"))
	labelNames = filenames.get("label_names")
	names = pickle.get("filenames")
	Y = np.array([0 for i in range(10*len(X.T))]).reshape(-1,10)
	c = 0
	for i in label:
		Y[c][i] = 1
		c+=1

	return X/255, label, labelNames, Y.T

def plotImage(x,name=""):
	x = x.reshape(3,32,32).transpose([1, 2, 0])
	plt.imshow(x, interpolation='gaussian')
	plt.title(name)
	plt.show()

def getMean(X):
	return np.mean(X,1).reshape(-1,1)

class Network():
	def __init__(self, setup, trainX, trainY, validationX, validationY, eta, batchSize = 200, regTerm = 0.1, p = 0.99, activationFunc = 'RELU', useBatch = False):
		self.W = []
		self.b = []
		for i in range(len(setup)-1):					#2/setup[i]
			self.W.append( np.array([np.random.normal(0,2/setup[i]) for j in range(setup[i]*setup[i+1])]).reshape(setup[i+1],setup[i]))									
			self.b.append( np.array([0.0 for i in range(setup[i+1])]).reshape(setup[i+1],1))
		self.p = p
		self.eta = eta
		self.batchSize = batchSize
		self.trainX = trainX
		self.trainY = trainY
		self.validationX = validationX
		self.validationY = validationY
		self.regTerm = regTerm
		self.WV = []
		self.bV = []
		for i in range(len(setup)-1):
			self.WV.append( np.array([0.0 for i in range(setup[i]*setup[i+1])]).reshape(setup[i+1],setup[i]))
			self.bV.append(np.array([0.0 for i in range(setup[i+1])]).reshape(setup[i+1],1))
		self.e = 1e-5
		self.activationFunc = activationFunc
		self.useBatch = useBatch

	def forwardPass(self, X):
		S = []
		S_ = []
		h = []
		mu = []
		v = []
		h.append(X)
		for l in range(len(self.W)-1):
			S.append((np.dot(self.W[l],h[l]) + self.b[l]))
			mul, vl = self.calculateNormalize(S[l],l)
			mu.append(mul)
			v.append(vl)
			S_.append(self.BatchNormalize(S[l],mul,vl,l))
			if self.useBatch:
				s = S_[l]
			else:
				s = S[l]
			if self.activationFunc == 'RELU':
				h.append(self.relu(s))

		S.append(np.dot(self.W[-1],h[-1]) + self.b[-1])
		return S, S_, mu, v, h, self.getP(S[-1])

	def evaluateClassifier(self, X):
		S, S_, mu, v, h, P = self.forwardPass(X)
		return P

	def computeAccuracy(self, X, Y):
		P = self.evaluateClassifier(X)
		acc = 0.0
		for sample in range(len(Y.T)):
			if np.argmax(P[:,sample]) == np.argmax(Y[:,sample]):
				acc+=1		
		return (acc/Y.shape[1])*100

	def computeCost(self,X, Y):
		P = self.evaluateClassifier(X)
		L = self.getL(P,Y)
		L /= X.shape[1]
		J = self.getJ(L)
		return J

	def sigmoid(self, X):
		return 1/(1+np.exp(-X))

	#Input is the hidden state
	def deltaSigmoid(self, X):
		return X * (1-X)

	def relu(self, h):
		return  np.where(h>0,h,0)

	def calculateNormalize(self, s, l):
		mu = np.mean(s,axis=1).reshape(-1,1)
		v = np.var(s,axis=1).reshape(-1,1)
		return mu, v
		


	def BatchNormalize(self, s, mu, v, l):
		diag = np.diagflat(v + self.e)
		left = np.power(diag,-0.5)
		ret = np.dot(left,s - mu)
		return ret

	def getP(self,S2):
		P = np.zeros((S2.shape[0],S2.shape[1]))
		for c in range(len(S2.T)):
			P[:,c] = np.exp(S2[:,c])/np.sum(np.exp(S2[:,c]))
		return P

	def getL(self, P, Y):
		total = 0.0
		for i in range(len(Y.T)):
			total += -np.log(np.dot(Y[:,i],P[:,i]))
		return total

	def getJ(self, L):
		return L + self.l2Req()

	def l2Req(self):
		total = 0.0
		for W in self.W:
			total += np.sum(W**2)
		#for w in np.nditer(self.W1):
		#	total+= w**2
		#for w in np.nditer(self.W2):
		#	total+= w**2
		return self.regTerm*total



	def updateWithBatch(self, X, Y):
		djdw = [np.zeros((self.W[i].shape)) for i in range(len(self.W))]
		djdb = [np.zeros(self.b[i].shape) for i in range(len(self.b))]
		
		addjdw, addjdb = self.calculateGradient(X,Y)
		for i in range(len(djdw)):
			djdw[i]+=addjdw[i]
			djdb[i]+=addjdb[i]
		for i in range(len(djdw)):
			djdw[i]/=X.shape[1]
			djdb[i]/=X.shape[1]

		for i in range(len(self.WV)):
			self.WV[i] = self.p * self.WV[i] + self.eta*djdw[i] + 2*self.regTerm*self.W[i]
		for i in range(len(self.bV)):
			self.bV[i] = self.p * self.bV[i] + self.eta*djdb[i]
		for i in range(len(self.W)):
			self.W[i] -= self.WV[i]
		for i in range(len(self.b)):
			self.b[i] -= self.bV[i]

	def batchNormBackPass(self, g, S, mu, v):
		Vb = []
		for b in range(len(v)):
			Vb.append(np.diagflat(v[b] + self.e))
		ds_idvb = -0.5 * np.dot(np.power(Vb,-(3/2)),np.diagflat(S - mu))




	def calculateGradient(self, x, y):
		djdw = [np.zeros((self.W[i].shape)) for i in range(len(self.W))]
		djdb = [np.zeros(self.b[i].shape) for i in range(len(self.b))]
		S, S_, mu, v, h, P = self.forwardPass(x)
		#backward
		g = P - y

		l = len(self.W)-1
		while l >= 0:
			djdb[l] = g
			djdw[l] = np.dot(g,h[l].T)
			g = np.dot(g.T,self.W[l])
			#HERE SHOULD BATCH NORM GO
			if l > 0:
				if self.activationFunc == 'RELU':
					ind = np.where(S[l-1]>0,1,0)

				for i in range(g.shape[0]):
					g[i,:] = np.dot(g[i,:],np.diag(ind[:,i]))
				g = g.T
			l-=1
		
		for i in range(len(djdw)):
			djdb[i] = np.sum(djdb[i], axis=1).reshape(-1,1)
		return djdw, djdb

	def fit(self, epochs=40, decayEta = False, earlyStopping = False):
		loss = []
		valLoss = []
		trainAcc = []
		validAcc = []
		for i in range(epochs):
			loss.append(self.computeCost(self.trainX, self.trainY))
			valLoss.append(self.computeCost(self.validationX, self.validationY))
			trainAcc.append(self.computeAccuracy(self.trainX, self.trainY)) 
			validAcc.append(self.computeAccuracy(self.validationX, self.validationY)) 
			if earlyStopping and i>5 and valLoss[i-1] - valLoss[i] < 1e-5:
				break
			for j in range(1,int(len(self.trainX.T)/self.batchSize)+1):
				jStart = (j-1)*self.batchSize
				jEnd = j * self.batchSize
				self.updateWithBatch(self.trainX[:,jStart:jEnd], self.trainY[:,jStart:jEnd])
			if decayEta and i % 10 == 0:
				self.eta = self.eta*0.95
			print(i)
		return loss, valLoss, trainAcc, validAcc

if __name__ == "__main__":
	#findThreeBestFromCoarse("sigmCoarse")
	#randomSearch("sigmCoarse", act='SIGM')

	#0.03	0.0001
	#0.03	0.00007
	#0.04	0.0001
	
	trainX, labelY, labelNames, trainY = getData("data_batch_1")
	trainX2, labelY2, labelNames2, trainY2 = getData("data_batch_2")
	trainX3, labelY3, labelNames3, trainY3 = getData("data_batch_3")
	trainX4, labelY4, labelNames4, trainY4 = getData("data_batch_4")
	trainX5, labelY5, labelNames5, trainY5 = getData("data_batch_5")
	testX, testLabelY, _, testY = getData("test_batch")

	trainX = np.concatenate((trainX, trainX2[:,0:9000], trainX3, trainX4, trainX5), axis=1)
	trainY = np.concatenate((trainY, trainY2[:,0:9000], trainY3, trainY4, trainY5), axis=1)
	validationX, valLabelY, _, validationY = getData("data_batch_2")

	#mean = getMean(trainX)
	#trainX = trainX - np.tile(mean,(1,trainX.shape[1]))
	#validationX = validationX - np.tile(mean, (1, validationX.shape[1]))
	#testX = testX - np.tile(mean, (1, testX.shape[1]))
	# PRETTY GOOD RELU
	eta = 0.001087721185986918
	lambd = 6.865912979562136e-07
	network = Network([3072, 50, 10], trainX, trainY, validationX, validationY, eta, regTerm=lambd, useBatch = False)
	loss, valLoss, trainAcc, validAcc = network.fit(epochs = 40, earlyStopping = False)

	plt.plot(loss, label="train loss")
	plt.plot(valLoss, label="validation loss")
	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.show()
	plt.plot(trainAcc, label="train accuracy")
	plt.plot(validAcc, label="validation accuracy")
	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.show()
	print(network.computeAccuracy(testX, testY))