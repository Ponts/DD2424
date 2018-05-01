import numpy as np
import copy
import lab2

def computeGradsNum(network, h=1e-5):

	grad_W1 = np.zeros((network.W1.shape[0], network.W1.shape[1]))
	grad_b1 = np.zeros((network.b1.shape[0], network.b1.shape[1]))
	grad_W2 = np.zeros((network.W2.shape[0], network.W2.shape[1]))
	grad_b2 = np.zeros((network.b2.shape[0], network.b2.shape[1]))

	c = network.computeCost(network.trainX, network.trainY)

	started_b = copy.deepcopy(network.b1)
	for i in range(len(network.b1)):
		b_try = np.copy(started_b)
		b_try[i] = b_try[i] + h
		network.b1 = np.copy(b_try)
		c2 = network.computeCost(network.trainX, network.trainY)
		grad_b1[i] = (c2-c)/h
		network.b1[i] = np.copy(started_b[i])

	network.b1 = started_b
	
	started_b = np.copy(network.b2)
	for i in range(len(network.b2)):
		b_try = np.copy(started_b)
		b_try[i] = b_try[i] + h
		network.b2 = np.copy(b_try)
		c2 = network.computeCost(network.trainX, network.trainY)
		grad_b2[i] = (c2-c)/h
		network.b2[i] = np.copy(started_b[i])

	network.b2 = started_b
	print("b done")


	started_W = np.copy(network.W1)
	for i in range(len(network.W1)):
		for j in range(len(network.W1.T)):
			W_try = np.copy(started_W)
			W_try[i][j] = W_try[i][j] + h
			network.W1 = np.copy(W_try)
			c2 = network.computeCost(network.trainX, network.trainY)
			grad_W1[i][j] = (c2-c)/h
			network.W1[i][j] = np.copy(started_W[i][j])
	
		print(i)
	network.W1 = started_W

	started_W = np.copy(network.W2)
	for i in range(len(network.W2)):
		for j in range(len(network.W2.T)):
			W_try = np.copy(started_W)
			W_try[i][j] = W_try[i][j] + h
			network.W2 = np.copy(W_try)
			c2 = network.computeCost(network.trainX, network.trainY)
			grad_W2[i][j] = (c2-c)/h
			network.W2[i][j] = np.copy(started_W[i][j])
	
		print(i)
	network.W2 = started_W
	return grad_W1, grad_b1, grad_W2, grad_b2



trainX, labelY, labelNames, trainY = lab2.getData("data_batch_1")
validationX, valLabelY, _, validationY = lab2.getData("data_batch_2")
testX, testLabelY, _, testY = lab2.getData("test_batch")
mean = lab2.getMean(trainX)
trainX = trainX - np.tile(mean,(1,trainX.shape[1]))
validationX = validationX - np.tile(mean, (1, validationX.shape[1]))
testX = testX - np.tile(mean, (1, testX.shape[1]))
end=1
trainX = trainX[:,0:end]
trainY = trainY[:,0:end]

network = lab2.TwoLayer([3072,50,10], trainX, trainY, validationX, validationY, 0.01, 100, regTerm=0.1, activationFunc = 'RELU')
e = 1e-6

grad_W1, grad_b1, grad_W2, grad_b2 = computeGradsNum(network, h=1e-6)

print("Formula difference in analytical and numerical:")
djdw1 = np.zeros((network.W1.shape[0], network.W1.shape[1]))
djdb1 = np.zeros((network.b1.shape[0], network.b1.shape[1]))
djdw2 = np.zeros((network.W2.shape[0], network.W2.shape[1]))
djdb2 = np.zeros((network.b2.shape[0], network.b2.shape[1]))
for i in range(len(trainX.T)):
	adddjdw1, adddjdb1, adddjdw2, adddjdb2 = network.calculateGradient(trainX[:,i:i+1],trainY[:,i:i+1])
	djdw1+=adddjdw1
	djdb1+=adddjdb1
	djdw2+=adddjdw2
	djdb2+=adddjdb2
djdw1/=trainX.shape[1]
djdb1/=trainX.shape[1]
djdw2/=trainX.shape[1]
djdb2/=trainX.shape[1]
analW1 = djdw1 + 2*network.regTerm*network.W1
analW2 = djdw2 + 2*network.regTerm*network.W2
analB1 = djdb1
analB2 = djdb2
print("W1")
print(np.max((abs(analW1 - grad_W1)) / np.clip((abs(grad_W1) + abs(analW1)), a_min=e, a_max=5000)))
print("B1")
print(np.max((abs(analB1 - grad_b1)) / np.clip((abs(grad_b1) + abs(analB1)), a_min=e, a_max=5000)))
print("W2")
print(np.max((abs(analW2 - grad_W2)) / np.clip((abs(grad_W2) + abs(analW2)), a_min=e, a_max=5000)))
print("B2")
print(np.max((abs(analB2 - grad_b2)) / np.clip((abs(grad_b2) + abs(analB2)), a_min=e, a_max=5000)))
print("done")

print("Absolute difference:")
print(np.max(abs(analW1-grad_W1)))
print(np.max(abs(analB1-grad_b1)))
print(np.max(abs(analW2-grad_W2)))
print(np.max(abs(analB2-grad_b2)))