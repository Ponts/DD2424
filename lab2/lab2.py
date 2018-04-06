import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pandas as pd
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
	def __init__(self, setup, trainX, trainY, validationX, validationY, eta, batchSize = 200, regTerm = 0.1, p = 0.99):
		self.W1 = np.array([np.random.normal(0,0.1) for i in range(setup[0]*setup[1])]).reshape(setup[1],setup[0])
		self.b1 = np.array([0.0 for i in range(setup[1])]).reshape(setup[1],1)
		self.W2 = np.array([np.random.normal(0,0.1) for i in range(setup[1]*setup[2])]).reshape(setup[2],setup[1])
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

	def getS1(self, X):
		return np.dot(self.W1,X) + self.b1

	def getH(self,S1):
		h = np.copy(S1)
		h[h < 0.0] = 0.0
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

		ind = np.where(S1>0,1,0)[:,0]
		#print(g.shape, ind.shape)
		g = np.dot(g,np.diag(ind)).T
		dldb1 = g
		dldw1 = np.dot(g,x.T)
		return dldw1, dldb1, dldw2, dldb2

	def fit(self, epochs=40, decayEta = False):
		accs = []
		valaccs = []
		accsies = []
		validaccsies = []
		testaccsies = []
		for i in range(epochs):
			accs.append(self.computeCost(self.trainX, self.trainY))
			valaccs.append(self.computeCost(self.validationX, self.validationY))
			accsies.append(self.computeAccuracy(self.trainX, self.trainY)) 
			validaccsies.append(self.computeAccuracy(self.validationX, self.validationY)) 
			testaccsies.append(self.computeAccuracy(testX, testY)) 
			#if i>0 and valaccs[i] > valaccs[i-1]:
			#	break
			for j in range(1,int(len(self.trainX.T)/self.batchSize)+1):
				jStart = (j-1)*self.batchSize
				jEnd = j * self.batchSize
				self.updateWithBatch(self.trainX[:,jStart:jEnd], self.trainY[:,jStart:jEnd])
			if decayEta:
				self.eta = self.eta*0.95
			print(i)
		return accs, valaccs, accsies#, validaccsies, testaccsies



trainX, labelY, labelNames, trainY = getData("data_batch_1")
validationX, valLabelY, _, validationY = getData("data_batch_2")
testX, testLabelY, _, testY = getData("test_batch")
mean = getMean(trainX)
trainX = trainX - np.tile(mean,(1,trainX.shape[1]))
validationX = validationX - np.tile(mean, (1, validationX.shape[1]))
'''
network = TwoLayer([3072, 50, 10], trainX, trainY, validationX, validationY, 0.01, 100, regTerm=0.000001, p = 0.95)
_,_,accs, validaccsies, testaccsies = network.fit(epochs=10)
plt.plot(accs, label="train accuracy")
plt.plot(validaccsies, label="validation accuracy")
plt.plot(testaccsies, label="test accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
print(max(validaccsies))
'''

bestEta = 0
bestLambda = 0.000001
bestTestAcc = 0.0
tries = 0
for eta in [0.5, 0.3, 0.1, 0.05, 0.01]:
	for lambd in [0.001, 0.005, 0.01, 0.05, 0.1]:
		#eta = np.random.uniform(low = etaRange[0], high = etaRange[1])
		#lambd = np.random.uniform(low=lRange[0], high=lRange[1])
		network = TwoLayer([3072, 50, 10], trainX, trainY, validationX, validationY, eta, 100, regTerm=lambd, p = 0.8)
		print("Eta: " + str(eta) + ", Lambda: " + str(lambd))

		accs, valaccs, accsies = network.fit(epochs=40, decayEta = True)
		testAcc = network.computeAccuracy(testX, testY)
		if testAcc > bestTestAcc:
			bestEta = eta
			bestTestAcc = testAcc
			bestLambda = lambd
		print(testAcc)
		valAcc = network.computeAccuracy(validationX, validationY)
		trainAcc = network.computeAccuracy(trainX, trainY)
		plt.plot(accs, label="eta: " + str(eta) + ", lambda: " + str(lambd))
		#plt.plot(valaccs,color="r", label="Validation loss")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		result = {'ValAccDone' : valAcc, 'trainAccDone' :  trainAcc, 'eta' : eta, 'lambda' : lambd, 'trainLoss' : accs, 'valLoss' : valaccs, 'trainAccuracy' : accsies}
		pd.DataFrame([result]).to_csv('coarseSearch/eta_' + str(eta) + '_lambda_' +str(lambd) + '.csv',  header=True)
		tries += 1
plt.legend()
plt.show()
print("BestEta: " + str(bestEta))
print("BestLamda: " + str(bestLambda))
print("bestTestAcc: " + str(bestTestAcc))

#mellan 0.05 och 0.15 verkar bra eta då p = 0.99
#samma för p = 0.5
#samma för p = 0.90
