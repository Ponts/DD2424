import numpy as np
import Ass1

def computeGradsNum(network, h=1e-5):
	no = network.W.shape[0]
	d = network.trainX.shape[0]

	grad_W = np.zeros((network.W.shape[0], network.W.shape[1]))
	grad_b = np.zeros((no, 1))

	c = network.computeCost(network.trainX, network.trainY)

	started_b = np.copy(network.b)
	for i in range(len(network.b)):
		b_try = np.copy(started_b)
		b_try[i] = b_try[i] + h
		network.b = np.copy(b_try)
		c2 = network.computeCost(network.trainX, network.trainY)
		grad_b[i] = (c2-c)/h
		network.b[i] = np.copy(started_b[i])

	network.b = started_b
	started_W = np.copy(network.W)
	print("b done")

	for i in range(len(network.W)):
		for j in range(len(network.W.T)):
			W_try = np.copy(started_W)
			W_try[i][j] = W_try[i][j] + h
			network.W = np.copy(W_try)
			c2 = network.computeCost(network.trainX, network.trainY)
			grad_W[i][j] = (c2-c)/h
			network.W[i][j] = np.copy(started_W[i][j])
	
		print(i)
	network.W = started_W
	return grad_W, grad_b

def computeGradsNumSlow(network, h=1e-5):
	no = network.W.shape[0]
	d = network.trainX.shape[0]

	grad_W = np.zeros((network.W.shape[0], network.W.shape[1]))
	grad_b = np.zeros((no, 1))

	started_b = np.copy(network.b)
	for i in range(len(network.b)):
		#print(network.b[0])
		b_try = np.copy(started_b)
		b_try[i] = b_try[i] - h
		network.b = np.copy(b_try)
		c1 = network.computeCost(network.trainX, network.trainY)
		b_try = np.copy(started_b)
		b_try[i] = b_try[i] + h
		network.b = np.copy(b_try)
		c2 = network.computeCost(network.trainX, network.trainY)
		grad_b[i] = (c2-c1)/(2*h)

	network.b = np.copy(started_b)
	started_W = np.copy(network.W)
	print("b done")

	for i in range(len(network.W)):
		for j in range(len(network.W.T)):
			W_try = np.copy(started_W)
			W_try[i][j] = W_try[i][j] - h
			network.W = np.copy(W_try)
			c1 = network.computeCost(network.trainX, network.trainY)
			W_try = np.copy(started_W)
			W_try[i][j] = W_try[i][j] + h
			network.W = np.copy(W_try)
			c2 = network.computeCost(network.trainX, network.trainY)
			grad_W[i][j] = (c2-c1)/(2*h)
	
		print(i)
	network.W = started_W
	return grad_W, grad_b

trainX, labelY, labelNames, trainY = Ass1.getData("data_batch_1")
validationX, valLabelY, _, validationY = Ass1.getData("data_batch_2")
testX, testLabelY, _, testY = Ass1.getData("test_batch")
end=10
trainX = trainX[:,0:end]
trainY = trainY[:,0:end]

network = Ass1.OneLayer(3072,10, trainX, trainY, validationX, validationY, 0.01, 100, regTerm=0.0)
e = 1e-6

grad_W, grad_b = computeGradsNum(network, h=1e-6)

print("Formula difference in analytical and numerical:")
analW = np.zeros((network.W.shape[0], network.W.shape[1]))
analB = np.zeros((network.b.shape[0], network.b.shape[1]))
for i in range(len(trainX[:,0:end].T)):
	adddjdw, adddjdb = network.calculateCEGradient(trainX[:,i:i+1],trainY[:,i:i+1])
	analW+=adddjdw
	analB+=adddjdb
analW/=trainX.shape[1]
analB/=trainY.shape[1]
analW += 2*network.regTerm*network.W
print("W")
print(np.min((abs(analW - grad_W)) / np.clip((abs(grad_W) + abs(analW)), a_min=e, a_max=5000)))
print("B")
print(np.max((abs(analB - grad_b)) / np.clip((abs(grad_b) + abs(analB)), a_min=e, a_max=5000)))
print("done")

print("Absolute difference:")
print(np.max(abs(analW-grad_W)))
print(np.max(abs(analB-grad_b)))