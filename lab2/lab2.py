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

class TwoLayer():
	def __init__(self, setup, trainX, trainY, validationX, validationY, eta, batchSize = 200, regTerm = 0.1, p = 0.99, activationFunc = 'RELU'):
		self.W1 = np.array([np.random.normal(0,2/setup[0]) for i in range(setup[0]*setup[1])]).reshape(setup[1],setup[0])
		self.b1 = np.array([0.0 for i in range(setup[1])]).reshape(setup[1],1)
		self.W2 = np.array([np.random.normal(0,2/setup[1]) for i in range(setup[1]*setup[2])]).reshape(setup[2],setup[1])
		self.b2 = np.array([0.0 for i in range(setup[2])]).reshape(setup[2],1)
		self.p = p
		self.eta = eta
		self.batchSize = batchSize
		self.trainX = trainX
		self.trainY = trainY
		self.validationX = validationX
		self.validationY = validationY
		self.regTerm = regTerm
		self.W1V = np.array([0.0 for i in range(setup[0]*setup[1])]).reshape(setup[1],setup[0])
		self.b1V = np.array([0.0 for i in range(setup[1])]).reshape(setup[1],1)
		self.W2V = np.array([0.0 for i in range(setup[1]*setup[2])]).reshape(setup[2],setup[1])
		self.b2V = np.array([0.0 for i in range(setup[2])]).reshape(setup[2],1)
		self.activationFunc = activationFunc


	def evaluateClassifier(self, X):
		S1 = self.getS1(X)
		h = self.getH(S1)
		s2 = self.getS2(h)
		return self.getP(s2)

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

	def getS1(self, X):
		return np.dot(self.W1,X) + self.b1

	def getH(self,S1):
		if self.activationFunc == 'RELU':
			h = np.copy(S1)
			h[h < 0.0] = 0.0
		elif self.activationFunc == 'SIGM':
			h = self.sigmoid(S1)
		return h

	def getS2(self, h):
		return np.dot(self.W2, h) + self.b2

	def getP(self,S2):
		P = np.zeros((S2.shape[0],S2.shape[1]))
		for c in range(len(S2.T)):
			P[:,c] = np.exp(S2[:,c])/sum(np.exp(S2[:,c]))
		return P

	def getL(self, P, Y):
		total = 0.0
		for i in range(len(Y.T)):
			total += -np.log(np.dot(Y[:,i],P[:,i]))
		return total

	def getJ(self, L):
		return L + self.l2Req()

	def l2Req(self):
		total = np.sum(self.W1**2) + np.sum(self.W2**2)
		#for w in np.nditer(self.W1):
		#	total+= w**2
		#for w in np.nditer(self.W2):
		#	total+= w**2
		return self.regTerm*total

	def updateWithBatch(self, X, Y):
		djdw1 = np.zeros((self.W1.shape[0], self.W1.shape[1]))
		djdb1 = np.zeros((self.b1.shape[0], self.b1.shape[1]))
		djdw2 = np.zeros((self.W2.shape[0], self.W2.shape[1]))
		djdb2 = np.zeros((self.b2.shape[0], self.b2.shape[1]))
		for i in range(len(X.T)):
			adddjdw1, adddjdb1, adddjdw2, adddjdb2 = self.calculateGradient(X[:,i:i+1],Y[:,i:i+1])
			djdw1+=adddjdw1
			djdb1+=adddjdb1
			djdw2+=adddjdw2
			djdb2+=adddjdb2
		djdw1/=X.shape[1]
		djdb1/=X.shape[1]
		djdw2/=X.shape[1]
		djdb2/=X.shape[1]
		self.W1V = self.p * self.W1V + self.eta*djdw1 + 2*self.regTerm*self.W1
		self.b1V = self.p * self.b1V + self.eta*djdb1
		self.W2V = self.p * self.W2V + self.eta*djdw2 + 2*self.regTerm*self.W2
		self.b2V = self.p * self.b2V + self.eta*djdb2
		self.W1 -= self.W1V
		self.b1 -= self.b1V
		self.W2 -= self.W2V
		self.b2 -= self.b2V

	def calculateGradient(self, x, y):
		S1 = self.getS1(x)
		h = self.getH(S1)
		s2 = self.getS2(h)
		P = self.getP(s2)
		g = P - y
		dldb2 = g
		dldw2 = np.dot(g,h.T)
		g = np.dot(g.T,self.W2)
		if self.activationFunc == 'RELU':
			ind = np.where(S1>0,1,0)[:,0]
			#print(g.shape, ind.shape)
			g = np.dot(g,np.diag(ind)).T
		elif self.activationFunc == 'SIGM':
			g = np.dot(g,self.deltaSigmoid(h))
		dldb1 = g
		dldw1 = np.dot(g,x.T)
		return dldw1, dldb1, dldw2, dldb2

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
			if decayEta:
				self.eta = self.eta*0.95
			print(i)
		return loss, valLoss, trainAcc, validAcc

def randomSearch(mode = "coarseSearch"):
	trainX, labelY, labelNames, trainY = getData("data_batch_1")
	validationX, valLabelY, _, validationY = getData("data_batch_2")
	testX, testLabelY, _, testY = getData("test_batch")
	mean = getMean(trainX)
	trainX = trainX - np.tile(mean,(1,trainX.shape[1]))
	validationX = validationX - np.tile(mean, (1, validationX.shape[1]))
	testX = testX - np.tile(mean, (1, testX.shape[1]))
	tries = 0
	while tries <150:
		eta = np.random.uniform(low = 0.025, high = 0.07)
		lambd = np.random.uniform(low=0.00005, high=0.002)
		network = TwoLayer([3072, 50, 10], trainX, trainY, validationX, validationY, eta, 100, regTerm=lambd, p = 0.9)
		print("Eta: " + str(eta) + ", Lambda: " + str(lambd))

		trainLoss, valLoss, trainAcc, valAcc = network.fit(epochs=20)
		#plt.plot(trainAcc, label="train accuracy")
		#plt.plot(valAcc, label="validation accuracy")
		#plt.legend()
		#plt.xlabel("Epochs")
		#plt.ylabel("Accuracy")
		#plt.show()
		testAcc = network.computeAccuracy(testX, testY)
		print(testAcc)
		valAccDone = network.computeAccuracy(validationX, validationY)
		print(valAccDone)
		#print(trainAcc[-1])
		result = {'ValAccDone' : valAccDone, 'trainAccDone' :  trainAcc, 'eta' : eta, 'lambda' : lambd, 'trainLoss' : trainLoss, 'valLoss' : valLoss, 'trainAccuracy' : trainAcc, 'valAccuracy' : valAcc}
		pd.DataFrame([result]).to_csv(mode + '/eta_' + str(eta) + '_lambda_' +str(lambd) + '.csv',  header=True)
		tries += 1



def findThreeBestFromCoarse(folder = "coarseSearch"):
	path =r'D:\\DD2424\\lab2\\' + folder
	allFiles = glob.glob(path + "/*.csv")
	frame = pd.DataFrame()
	list_ = []
	for file_ in allFiles:
	    df = pd.read_csv(file_,index_col=None)#, header=0)
	    list_.append(df)
	frame = pd.concat(list_)
	nr = 3
	bestThree = [0.0] * nr
	bestThreeI = [0] * nr
	index = 0
	for _, row in frame.iterrows():
		withHypers = np.max(eval(row['valAccuracy']))
		for i in range(len(bestThree)):
			if withHypers > bestThree[i]:
				bestThree = bestThree[:i] + [withHypers] + bestThree[i:-1]
				bestThreeI = bestThreeI[:i] + [index] + bestThreeI[i:-1]
				break
		index+=1
	print(bestThreeI)
	print("BestSettings:")
	for index in bestThreeI:
		print(frame.eta.values[index], frame['lambda'].values[index], frame['ValAccDone'].values[index])
	for index in bestThreeI:
		plt.plot(eval(frame.valAccuracy.values[index]), label="eta:" + str(frame.eta.values[index]) + " lambda: " + str(frame['lambda'].values[index]))
		#plt.plot(eval(frame.valLoss.values[index]), label = "validation loss")
		
	plt.legend()
	plt.show()
	print("Validation Accuracy: ")
	for index in bestThreeI:
		print(np.max(eval(frame.valAccuracy.values[index])))

	


if __name__ == "__main__":
	#findThreeBestFromCoarse("coarseSearch")
	#randomSearch("fineSearch")

	
	trainX, labelY, labelNames, trainY = getData("data_batch_1")
	trainX2, labelY2, labelNames2, trainY2 = getData("data_batch_2")
	trainX3, labelY3, labelNames3, trainY3 = getData("data_batch_3")
	trainX4, labelY4, labelNames4, trainY4 = getData("data_batch_4")
	trainX5, labelY5, labelNames5, trainY5 = getData("data_batch_5")
	testX, testLabelY, _, testY = getData("test_batch")

	trainX = np.concatenate((trainX, trainX2[:,0:9000], trainX3, trainX4, trainX5), axis=1)
	trainY = np.concatenate((trainY, trainY2[:,0:9000], trainY3, trainY4, trainY5), axis=1)
	validationX, valLabelY, _, validationY = getData("data_batch_2")

	mean = getMean(trainX)
	trainX = trainX - np.tile(mean,(1,trainX.shape[1]))
	validationX = validationX - np.tile(mean, (1, validationX.shape[1]))
	testX = testX - np.tile(mean, (1, testX.shape[1]))
	# PRETTY GOOD
	eta = 0.03244093510324865
	lambd = 0.00011369581140139731
	#
	network = TwoLayer([3072, 50, 10], trainX, trainY, validationX[:,9000:-1], validationY[:,9000:-1], eta, 100, regTerm=lambd, p = 0.9, activationFunc = 'RELU')
	trainLoss, valLoss, trainAcc, valAcc = network.fit(epochs=200, decayEta = False, earlyStopping = True)
	plt.plot(trainLoss, label="train loss")
	plt.plot(valLoss, label="validation loss")
	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.show()
	plt.plot(trainAcc, label="train accuracy")
	plt.plot(valAcc, label="validation accuracy")
	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.show()
	print(network.computeAccuracy(testX, testY))
	



#mellan 0.001 och 0.15 verkar bra eta 
#mellan 0.0001 och 0.001


#FINE SEARCH
#eta (low = 0.025, high = 0.07)
#lambda (low=0.00005, high=0.002)