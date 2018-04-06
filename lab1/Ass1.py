import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
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


class OneLayer:
	def __init__(self, inputs, outputs, trainX, trainY, validationX, validationY, eta, batchSize = 200, regTerm = 0.1):
		self.W = np.array([np.random.normal(0,0.1) for i in range(inputs*outputs)]).reshape(outputs,inputs)
		self.b = np.array([np.random.normal(0,0.1) for i in range(outputs)]).reshape(outputs,1)
		self.eta = eta
		self.batchSize = batchSize
		self.trainX = trainX
		self.trainY = trainY
		self.validationX = validationX
		self.validationY = validationY
		self.regTerm = regTerm

	def evaluateClassifier(self, X):
		Z = self.getZ(X)
		S = self.getS(Z)
		return self.getP(S)

	def computeCost(self,X, Y):
		P = self.evaluateClassifier(X)
		L = self.getL(P,Y)
		L /= X.shape[1]
		J = self.getJ(L)
		return J

	def computeSVMCost(self,X,Y):
		Z = self.getZ(X)
		S = self.getS(Z)
		L = 0.0
		for sample in range(len(Y.T)):
			chosenY = np.argmax(Y[:,sample])
			#print(S[0][sample])
			rightTerm = [max(0,S[i,sample] - S[chosenY,sample] + 1) for i in range(len(S))]
			rightTerm[chosenY] = 0.0
			L += sum([max(0,score) for score in rightTerm])
		L /= X.shape[1]
		return L
		

	def computeAccuracy(self, X, Y):
		P = self.evaluateClassifier(X)
		acc = 0.0
		for sample in range(len(Y.T)):
			if np.argmax(P[:,sample]) == np.argmax(Y[:,sample]):
				acc+=1		
		return (acc/Y.shape[1])*100

	def calculateCEGradient(self,x,y):
		P = self.evaluateClassifier(x)
		g = P - y
		dldb = g
		dldw = np.dot(g,x.T)
		return dldw, dldb

	def calculateSVMGradient(self,x,y):
		Z = self.getZ(x)
		S = self.getS(Z)
		chosenY = np.argmax(y)
		dlds = np.array([0.0 if S[i] - S[chosenY] + 1 < 0 else 1.0 for i in range(len(S))]).reshape(-1,1)
		dlds[chosenY] = sum([0 if S[i] - S[chosenY] + 1 < 0 and i != chosenY else -1 for i in range(len(S)) ])
		dldw = np.dot(dlds,x.T)
		return dldw, dlds

	def updateWithBatch(self,X,Y):
		djdw = np.zeros((self.W.shape[0], self.W.shape[1]))
		djdb = np.zeros((self.b.shape[0], self.b.shape[1]))
		for i in range(len(X.T)):
			adddjdw, adddjdb = self.calculateCEGradient(X[:,i:i+1],Y[:,i:i+1])
			djdw+=adddjdw
			djdb+=adddjdb
		djdw/=X.shape[1]
		djdb/=X.shape[1]
		self.W -= self.eta*(djdw + 2*self.regTerm*self.W)
		self.b -= self.eta*djdb

	def updateWithBatchSVM(self,X,Y):
		djdw = np.zeros((self.W.shape[0], self.W.shape[1]))
		djdb = np.zeros((self.b.shape[0], self.b.shape[1]))
		for i in range(len(X.T)):
			adddjdw, adddjdb = self.calculateSVMGradient(X[:,i:i+1],Y[:,i:i+1])
			djdw1+=adddjdw
			djdb1+=adddjdb
		djdw/=X.shape[1]
		djdb/=X.shape[1]
		self.W -= self.eta*(djdw + 2*self.regTerm*self.W)
		self.b -= self.eta*djdb

	def getZ(self,x):
		return np.dot(self.W,x)

	def getS(self,Z):
		S = (Z+self.b)
		return S

	def getP(self,S):
		P = np.zeros((S.shape[0],S.shape[1]))
		for c in range(len(S.T)):
			P[:,c] = np.exp(S[:,c])/np.sum(np.exp(S[:,c]))
		return P

	def getL(self,P,Y):
		total = 0.0
		for i in range(len(Y.T)):
			total += -np.log(np.dot(Y[:,i],P[:,i]))
		return total

	def getJ(self,L):
		return L + self.l2Req()

	def l2Req(self):
		total = np.sum(self.W**2)
		#total = 0.0
		#for w in np.nditer(self.W):
		#	total += w**2
		return self.regTerm*total

	def forward(self,x,y):
		Z = self.getZ(x)
		S = self.getS(Z)
		P = self.getP(S)
		L = self.getL(P,y)
		J = self.getJ(L)
		return P,L,J

	def fit(self, epochs=40, decayEta = False, saveBestModel=False):
		if saveBestModel:
			bestW = self.W
			bestB = self.b
			bestValAcc = 0.0
		accs = [0.0]*epochs
		valaccs = [0.0]*epochs
		for i in range(epochs):
			accs[i] = self.computeCost(self.trainX, self.trainY)
			valaccs[i] = self.computeCost(self.validationX, self.validationY)
			for j in range(1,int(len(self.trainX.T)/self.batchSize)+1):
				jStart = (j-1)*self.batchSize
				jEnd = j * self.batchSize
				self.updateWithBatch(self.trainX[:,jStart:jEnd], self.trainY[:,jStart:jEnd])
			if decayEta:
				self.eta = self.eta*0.9
			newAcc = self.computeAccuracy(self.validationX, self.validationY) 
			if saveBestModel and newAcc > bestValAcc:
				bestW = np.copy(self.W)
				bestB = np.copy(self.b)
				bestValAcc = newAcc
			print (i)
		if saveBestModel:
			self.W = np.copy(bestW)
			self.b = np.copy(bestB)
		return accs, valaccs

	def fitSVM(self, epochs=40, decayEta = False, saveBestModel=False):
		if saveBestModel:
			bestW = self.W
			bestB = self.b
			bestValAcc = 0.0
		accs = [0.0]*epochs
		valaccs = [0.0]*epochs
		for i in range(epochs):
			accs[i] = self.computeSVMCost(self.trainX, self.trainY)
			valaccs[i] = self.computeSVMCost(self.validationX, self.validationY)
			for j in range(1,int(len(self.trainX.T)/self.batchSize)+1):
				jStart = (j-1)*self.batchSize
				jEnd = j * self.batchSize
				self.updateWithBatchSVM(self.trainX[:,jStart:jEnd], self.trainY[:,jStart:jEnd])
			if decayEta:
				self.eta = self.eta*0.9
			newAcc = self.computeAccuracy(self.validationX, self.validationY)
			if saveBestModel and newAcc > bestValAcc:
				print("New best found woho")
				bestW = np.copy(self.W)
				bestB = np.copy(self.b)
				bestValAcc = newAcc
			print(i)
		if saveBestModel:
			self.W = np.copy(bestW)
			self.b = np.copy(bestB)
		return accs, valaccs


	def printWeights(self):
		fig, axes = plt.subplots(1,10)
		_,_,labelNames,_ = getData("data_batch_1")

		for i in range(len(axes)):
			pic = self.W[i].reshape(3,32,32).transpose([1, 2, 0])
			pic = (pic - pic.min()) / (pic.max() - pic.min())
			#pic = pic.T
			axes[i].imshow(pic, interpolation="gaussian")
			axes[i].set_title(labelNames[i])
			axes[i].set_xticks(())
			axes[i].set_yticks(())
		plt.show()


def noImprovementsTest():
	trainX, labelY, labelNames, trainY = getData("data_batch_1")
	validationX, valLabelY, _, validationY = getData("data_batch_2")
	testX, testLabelY, _, testY = getData("test_batch")
	network = OneLayer(3072,10, trainX, trainY, validationX, validationY, 0.01, 100, regTerm=0.0)
	epochs=40
	accs, valaccs = network.fit(epochs, saveBestModel=True)
	plt.plot(accs, color="g", label="Training loss")
	plt.plot(valaccs,color="r", label="Validation loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	#plt.show()
	#network.printWeights()
	print(network.computeAccuracy(testX, testY))
	print(network.computeAccuracy(validationX, validationY))
	return network

def withImprovementsTest():
	trainX, labelY, labelNames, trainY = getData("data_batch_1")
	trainX2, labelY2, labelNames2, trainY2 = getData("data_batch_2")
	trainX3, labelY3, labelNames3, trainY3 = getData("data_batch_3")
	trainX4, labelY4, labelNames4, trainY4 = getData("data_batch_4")
	trainX5, labelY5, labelNames5, trainY5 = getData("data_batch_5")
	testX, testLabelY, _, testY = getData("test_batch")

	trainX = np.concatenate((trainX,trainX2, trainX3, trainX4, trainX5), axis=1)
	trainY = np.concatenate((trainY,trainY2, trainY3, trainY4, trainY5), axis=1)

	validationX, valLabelY, _, validationY = getData("data_batch_2")
	network = OneLayer(3072,10, trainX, trainY, validationX, validationY, 0.01, 100, regTerm=0.0001)
	epochs=40

	accs, valaccs = network.fit(epochs, decayEta = False, saveBestModel=True)

	plt.plot(accs, color="g", label="Training loss")
	plt.plot(valaccs,color="r", label="Validation loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	#plt.show()
	#network.printWeights()

	testAcc = (network.computeAccuracy(testX, testY))
	print(testAcc)
	valAcc = network.computeAccuracy(validationX, validationY)
	print(valAcc)
	return network

def testSVMLoss():
	trainX, labelY, labelNames, trainY = getData("data_batch_1")
	validationX, valLabelY, _, validationY = getData("data_batch_2")
	testX, testLabelY, _, testY = getData("test_batch")
	network = OneLayer(3072,10, trainX, trainY, validationX, validationY, 0.005, 100, regTerm=0.01)
	epochs=40
	accs, valaccs = network.fitSVM(epochs, saveBestModel=True)
	plt.plot(accs, color="g", label="Training loss")
	plt.plot(valaccs,color="r", label="Validation loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()
	network.printWeights()
	print(network.computeAccuracy(testX, testY))
	print(network.computeAccuracy(validationX, validationY))

def checkEnsembleAccuracy(votes, Y):
	acc = 0.0
	for sample in range(len(Y.T)):
		if votes[sample] == np.argmax(Y[:,sample]):
			acc+=1		
	return (acc/Y.shape[1])*100
	


#X är rätt
#Y är rätt
#W är rätt
#b är rätt
#P är rätt
# K = 10, N = 1000, D = 3072
testX, testLabelY, _, testY = getData("test_batch")
networks = []
for i in range(10):
	networks.append(withImprovementsTest())

Ps = np.array([network.evaluateClassifier(testX) for network in networks])
votes = np.zeros((len(networks), Ps.shape[2]), dtype=np.int8)
for n in range(len(networks)):
	for sample in range(Ps.shape[2]):
		votes[n][sample] = int(np.argmax(Ps[n,:,sample]))


ensembleVote = np.array([np.argmax(np.bincount(votes[:,sample])) for sample in range(votes.shape[1])])
print("testAccuracy: ")
print(checkEnsembleAccuracy(ensembleVote, testY))
'''
bestAcc = 0
for i in range(10):
	acc = withImprovementsTest()
	if bestAcc < acc:
		bestAcc = acc
print("Best: " + str(bestAcc))
'''
#testSVMLoss()

#Best without improvements:  				34.39 % validation, 39.33 train

#Ensemble vote:								36.26 %

#Added training on all data: 				40.68 %

#Decaying eta on all data: 					38.05 %

#Decaying eta:								34.46 %

#Saving best model:							35.22 %

#Saving best model and all data:			41.39 %

#Saving best model, all data and decay eta:	41.28 %

#Saving best model, all data, ensemble: 	41.31 %	

#index = np.random.randint(0,len(trainX.T))
#plotImage(trainX[:,index],labelNames[labelY[index]])