import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pandas as pd
import glob
import copy
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
	def __init__(self, setup, trainX, trainY, validationX, validationY, eta, batchSize = 200, regTerm = 0.1, p = 0.99, activationFunc = 'RELU', useBatch = False, alpha=0.99):
		self.W = []
		self.b = []
		for i in range(len(setup)-1):					#2/setup[i]
			self.W.append( np.array([np.random.normal(0,2/setup[i]) for k in range(setup[i]*setup[i+1])]).reshape(setup[i+1],setup[i]))									
			self.b.append( np.array([0.0 for k in range(setup[i+1])]).reshape(setup[i+1],1))
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
			self.WV.append( np.array([0.0 for k in range(setup[i]*setup[i+1])]).reshape(setup[i+1],setup[i]))
			self.bV.append(np.array([0.0 for k in range(setup[i+1])]).reshape(setup[i+1],1))
		self.e = 1e-5
		self.activationFunc = activationFunc
		self.useBatch = useBatch
		self.muav = []
		self.vav = []
		for l in range(1,len(setup)-1):
			self.muav.append( np.zeros((setup[l],1) ))
			self.vav.append(   np.zeros((setup[l],1) )) 
		self.alpha = alpha
		self.first = True

	def forwardPass(self, X):
		S = []
		S_ = []
		h = []
		mu = []
		v = []
		h.append(X)
		for l in range(len(self.W)-1):
			S.append((np.dot(self.W[l],h[l]) + self.b[l]))
			if self.useBatch:
				mul, vl = self.calculateNormalize(S[l])
				mu.append(mul)
				v.append(vl)

				S_.append(self.BatchNormalize(S[l],mul,vl))
				s = S_[l]
			else:
				s = S[l]
			if self.activationFunc == 'RELU':
				h.append(self.relu(s))
			elif self.activationFunc == 'SIGM':
				h.append(self.sigmoid(s))

		S.append(np.dot(self.W[-1],h[-1]) + self.b[-1])
		return S, S_, mu, v, h, self.getP(S[-1])

	def testPass(self, X):
		S = []
		S_ = []
		h = []
		h.append(X)
		for l in range(len(self.W)-1):
			S.append((np.dot(self.W[l],h[l]) + self.b[l]))
			if self.useBatch:
				S_.append(self.BatchNormalize(S[l],self.muav[l],self.vav[l]))
				s = S_[l]
			else:
				s = S[l]
			if self.activationFunc == 'RELU':
				h.append(self.relu(s))
			elif self.activationFunc == 'SIGM':
				h.append(self.sigmoid(s))
		S.append(np.dot(self.W[-1],h[-1]) + self.b[-1])
		return self.getP(S[-1])

	def evaluateClassifier(self, X):
		S, S_, mu, v, h, P = self.forwardPass(X)
		return P

	def computeAccuracy(self, X, Y):
		P = self.testPass(X)
		acc = 0.0
		for sample in range(len(Y.T)):
			if np.argmax(P[:,sample]) == np.argmax(Y[:,sample]):
				acc+=1
		return (acc/Y.shape[1])*100

	def computeCost(self,X, Y):
		P = self.testPass(X)
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

	def calculateNormalize(self, s):
		mu = np.mean(s,axis=1).reshape(-1,1)
		v = np.mean((s - mu)**2, axis=1).reshape(-1,1)
		return mu, v
		
	def BatchNormalize(self, s, mu, v):
		vb = np.power(v + self.e,-0.5)
		diag = np.diagflat(vb)
		ret = np.dot(diag,s - mu)
		
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
			self.bV[i] = self.p * self.bV[i] + self.eta*djdb[i]
			self.W[i] -= self.WV[i]
			self.b[i] -= self.bV[i]

	def batchNormBackPass(self, g, S, mu, v):
		Vb = v + self.e
		#print("---------")
		#print(Vb)
		vBSq = np.power(Vb, -0.5)
		#print(vBSq)
		sMu = S - mu
		n = S.shape[1]
		gVgSq = np.multiply(g, vBSq)
		gradVar = -0.5 * np.sum(np.multiply(np.multiply(g,np.power(Vb, -3./2)), sMu), axis=1).reshape(-1,1)
		gradMu = - np.sum(gVgSq, axis=1).reshape(-1,1)
		return gVgSq + (2/n) * np.multiply(gradVar, sMu) + gradMu/n

	def calculateGradient(self, x, y):
		djdw = [np.zeros((self.W[i].shape)) for i in range(len(self.W))]
		djdb = [np.zeros(self.b[i].shape) for i in range(len(self.b))]
		S, S_, mu, v, h, P = self.forwardPass(x)
		if self.first and self.useBatch:
			self.muav = copy.deepcopy(mu)
			self.vav = copy.deepcopy(v)
		elif self.useBatch:
			for l in range(len(self.muav)):
				self.muav[l] = self.alpha * self.muav[l] + (1.-self.alpha)*mu[l]
				self.vav[l] = self.alpha * self.vav[l] + (1.-self.alpha)*v[l]
		#backward
		g = P - y
		l = len(self.W)-1
		while l >= 0:
			djdb[l] = g
			djdw[l] = np.dot(g,h[l].T)
			g = np.dot(self.W[l].T,g)
			
			if l > 0:
				if self.useBatch:
					s = S_[l-1]
				else:
					s = S[l-1]
				if self.activationFunc == 'RELU':
					ind = np.where(s>0.,1.,0.)
				g = np.multiply(g,ind)
				if self.useBatch:
					g= self.batchNormBackPass(g,S[l-1],mu[l-1],v[l-1])
			l-=1
		
		for i in range(len(djdw)):
			djdb[i] = np.sum(djdb[i], axis=1).reshape(-1,1)
		return djdw, djdb

	def fit(self, epochs=40, decayEta = False, earlyStopping = False):
		loss = []
		valLoss = []
		trainAcc = []
		validAcc = []
		self.first = True
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
				self.first = False
				
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

	

	validationX, valLabelY, _, validationY = getData("data_batch_2")
	validationX = validationX[:,9000:]
	validationY = validationY[:,9000:]
	trainX = np.concatenate((trainX, trainX2[:,0:9000], trainX3, trainX4, trainX5), axis=1)
	trainY = np.concatenate((trainY, trainY2[:,0:9000], trainY3, trainY4, trainY5), axis=1)
	mean = getMean(trainX)
	trainX = trainX - np.tile(mean,(1,trainX.shape[1]))
	validationX = validationX - np.tile(mean, (1, validationX.shape[1]))
	testX = testX - np.tile(mean, (1, testX.shape[1]))
	# PRETTY GOOD RELU
	eta = 0.00001
	lambd = 0.000000000
	network = Network([3072, 50, 10], trainX, trainY, validationX, validationY, eta, regTerm=lambd, activationFunc='RELU', useBatch = False)
	loss, valLoss, trainAcc, validAcc = network.fit(epochs = 50, earlyStopping = True)

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
	plt.grid(True)
	plt.show()
	print(network.computeAccuracy(testX, testY))