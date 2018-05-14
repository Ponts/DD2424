import numpy as np
import copy
import lab3

def computeGradsNum(network, h=1e-5):
	grad_W = [np.zeros((network.W[i].shape)) for i in range(len(network.W))]
	grad_b = [np.zeros(network.b[i].shape) for i in range(len(network.b))]

	c = network.computeCost(network.trainX, network.trainY, num=network.useBatch)

	for l in range(len(network.b)):
		started_b = copy.deepcopy(network.b[l])
		for i in range(len(network.b[l])):
			b_try = np.copy(started_b)
			b_try[i] = b_try[i] + h
			network.b[l] = np.copy(b_try)
			c2 = network.computeCost(network.trainX, network.trainY, num=network.useBatch)
			grad_b[l][i] = (c2-c)/h
			network.b[l][i] = np.copy(started_b[i])
		network.b[l] = started_b
	print("b done")

	for l in range(len(network.W)):
		started_W = np.copy(network.W[l])
		for i in range(len(network.W[l])):
			for j in range(len(network.W[l].T)):
				W_try = np.copy(started_W)
				W_try[i][j] = W_try[i][j] + h
				network.W[l] = np.copy(W_try)
				c2 = network.computeCost(network.trainX, network.trainY, num=network.useBatch)
				grad_W[l][i][j] = (c2-c)/h
				network.W[l][i][j] = np.copy(started_W[i][j])
			print(i)
		network.W[l] = started_W
	return grad_W, grad_b

def computeGradsNumSlow(network, h=1e-6):

	grad_W = [np.zeros((network.W[i].shape)) for i in range(len(network.W))]
	grad_b = [np.zeros(network.b[i].shape) for i in range(len(network.b))]

	for l in range(len(network.b)):
		started_b = copy.deepcopy(network.b[l])
		for i in range(len(network.b[l])):
			b_try = np.copy(started_b)
			b_try[i] = b_try[i] - h
			network.b[l] = np.copy(b_try)
			c1 = network.computeCost(network.trainX, network.trainY, num=network.useBatch)
			b_try = np.copy(started_b)
			b_try[i] += h
			network.b[l] = np.copy(b_try)
			c2 = network.computeCost(network.trainX, network.trainY, num=network.useBatch)
			b_try = np.copy(started_b)
			grad_b[l][i] = (c2-c1)/(2*h)
			network.b[l][i] = np.copy(started_b[i])
		network.b[l] = started_b
	print("b done")

	for l in range(len(network.W)):
		started_W = np.copy(network.W[l])
		for i in range(len(network.W[l])):
			for j in range(len(network.W[l].T)):
				W_try = np.copy(started_W)
				W_try[i][j] = W_try[i][j] - h
				network.W[l] = np.copy(W_try)
				c1 = network.computeCost(network.trainX, network.trainY, num=network.useBatch)
				W_try = np.copy(started_W)
				W_try[i][j] += h
				network.W[l] = np.copy(W_try)
				c2 = network.computeCost(network.trainX, network.trainY, num=network.useBatch)
				W_try = np.copy(started_W)
				grad_W[l][i][j] = (c2-c1)/(2*h)
				network.W[l][i][j] = np.copy(started_W[i][j])
			print(i)
		network.W[l] = started_W
	return grad_W, grad_b



trainX, labelY, labelNames, trainY = lab3.getData("data_batch_1")
validationX, valLabelY, _, validationY = lab3.getData("data_batch_2")
testX, testLabelY, _, testY = lab3.getData("test_batch")
mean = lab3.getMean(trainX)
trainX = trainX - np.tile(mean,(1,trainX.shape[1]))
validationX = validationX - np.tile(mean, (1, validationX.shape[1]))
testX = testX - np.tile(mean, (1, testX.shape[1]))
end=3
trainX = trainX[:,0:end]
trainY = trainY[:,0:end]
eta = 0.1
lambd = 0.
network = network = lab3.Network([3072, 50, 50, 30, 10], trainX, trainY, validationX, validationY, eta, regTerm=lambd, activationFunc='RELU', useBatch = True)
for l in range(len(network.muav)):
	network.muav[l] = ( np.zeros(network.muav[l].shape ))
	network.vav[l] = (   np.zeros(network.vav[l].shape )) 
e = 1e-6

grad_W, grad_b = computeGradsNumSlow(network, h=1e-6)
for l in range(len(network.muav)):
	network.muav[l] = ( np.zeros(network.muav[l].shape ))
	network.vav[l] = (   np.zeros(network.vav[l].shape ))



print("Formula difference in analytical and numerical:")
djdw = [np.zeros((network.W[i].shape)) for i in range(len(network.W))]
djdb = [np.zeros(network.b[i].shape) for i in range(len(network.b))]
analW = []
analB = []
addjdw, addjdb = network.calculateGradient(trainX,trainY)
for l in range(len(djdw)):
	djdw[l]+=addjdw[l]
	djdb[l]+=addjdb[l]
for l in range(len(djdw)):
	djdw[l]/=trainX.shape[1]
	djdb[l]/=trainX.shape[1]
for l in range(len(network.W)):
	analW.append( djdw[l] + 2*network.regTerm*network.W[l])
	analB.append(djdb[l])

for l in range(len(network.W)):
	print("-----")
	print(np.max(grad_W[l]))
	print(np.max(analW[l]))
print("W")
for l in range(len(network.W)):
	print(np.max((abs(analW[l] - grad_W[l])) / np.clip((abs(grad_W[l]) + abs(analW[l])), a_min=e, a_max=5000)))
print("B")
for l in range(len(network.b)):
	print(np.max((abs(analB[l] - grad_b[l])) / np.clip((abs(grad_b[l]) + abs(analB[l])), a_min=e, a_max=5000)))

