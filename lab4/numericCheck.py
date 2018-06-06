import lab4
import reader
import numpy as np

def computeGradsNumSlow(network, x, y, h=1e-5):

	grad_b = np.zeros((network.b.shape))
	grad_W = np.zeros((network.W.shape))
	grad_V = np.zeros((network.V.shape))
	grad_U = np.zeros((network.U.shape))
	grad_c = np.zeros((network.c.shape))

	started_b = np.copy(network.b)
	for i in range(len(network.b)):
		b_try = np.copy(started_b)
		b_try[i] = b_try[i] - h
		network.b = np.copy(b_try)
		c1 = network.calculateLoss(x, y)
		b_try = np.copy(started_b)
		b_try[i] = b_try[i] + h
		network.b = np.copy(b_try)
		c2 = network.calculateLoss(x, y)
		grad_b[i] = (c2-c1)/(2*h)

	network.b = np.copy(started_b)
	started_W = np.copy(network.W)
	print("b done")
	
	for i in range(len(network.W)):
		for j in range(len(network.W.T)):
			W_try = np.copy(started_W)
			W_try[i][j] = W_try[i][j] - h
			network.W = np.copy(W_try)
			c1 = network.calculateLoss(x, y)
			W_try = np.copy(started_W)
			W_try[i][j] = W_try[i][j] + h
			network.W = np.copy(W_try)
			c2 = network.calculateLoss(x, y)
			grad_W[i][j] = (c2-c1)/(2*h)
		print(i)
		
	network.W = np.copy(started_W)
	print("W done")
	
	started_V = np.copy(network.V)
	
	for i in range(len(network.V)):
		for j in range(len(network.V.T)):
			V_try = np.copy(started_V)
			V_try[i][j] = V_try[i][j] - h
			network.V = np.copy(V_try)
			c1 = network.calculateLoss(x, y)
			V_try = np.copy(started_V)
			V_try[i][j] = V_try[i][j] + h
			network.V = np.copy(V_try)
			c2 = network.calculateLoss(x, y)
			grad_V[i][j] = (c2-c1)/(2*h)
	
		print(i)
	
	print("V done")
	network.V = np.copy(started_V)
	
	started_U = np.copy(network.U)
	for i in range(len(network.U)):
		for j in range(len(network.U.T)):
			U_try = np.copy(started_U)
			U_try[i][j] = U_try[i][j] - h
			network.U = np.copy(U_try)
			c1 = network.calculateLoss(x, y)
			U_try = np.copy(started_U)
			U_try[i][j] = U_try[i][j] + h
			network.U = np.copy(U_try)
			c2 = network.calculateLoss(x, y)
			grad_U[i][j] = (c2-c1)/(2*h)
	
		print(i)

	print("U done")
	network.U = np.copy(network.U)

	
	started_c = np.copy(network.c)
	for i in range(len(network.c)):
		c_try = np.copy(started_c)
		c_try[i] = c_try[i] - h
		network.c = np.copy(c_try)
		c1 = network.calculateLoss(x, y)
		c_try = np.copy(started_c)
		c_try[i] = c_try[i] + h
		network.c = np.copy(c_try)
		c2 = network.calculateLoss(x, y)
		grad_c[i] = (c2-c1)/(2*h)

	network.c = np.copy(started_c)
	

	return grad_W, grad_b, grad_V, grad_U, grad_c


dh = reader.DataHandler()
print("data aquired")
network = lab4.Network([dh.len,100,dh.len])
x,y = dh.getInputOutput(0,25)

grad_W, grad_b, grad_V, grad_U, grad_c = computeGradsNumSlow(network, x, y)
e = 1e-6
dldw, dldb, dldu, dldv, dldc = network.calculateGradient(x,y)

print("W num")
print(np.mean(grad_W))
print("b num")
print(np.mean(grad_b))
print("U num")
print(np.mean(grad_U))
print("----------")
print("W anal")
print(np.mean(dldw))
print("b anal")
print(np.mean(dldb))
print("U anal")
print(np.mean(dldu))

print("W")
print(np.max((abs(dldw - grad_W)) / np.clip((abs(grad_W) + abs(dldw)), a_min=e, a_max=9999)))
print("B")
print(np.max((abs(dldb - grad_b)) / np.clip((abs(grad_b) + abs(dldb)), a_min=e, a_max=9999)))
print("V")
print(np.max((abs(dldv - grad_V)) / np.clip((abs(grad_V) + abs(dldv)), a_min=e, a_max=9999)))
print("U")
print(np.max((abs(dldu - grad_U)) / np.clip((abs(grad_U) + abs(dldu)), a_min=e, a_max=9999)))
print("c")
print(np.max((abs(dldc - grad_c)) / np.clip((abs(grad_c) + abs(dldc)), a_min=e, a_max=9999)))
print("done")

print("Absolute difference:")
print(np.max(abs(dldw-grad_W)))
print(np.max(abs(dldb-grad_b)))
print(np.max(abs(dldv-grad_V)))
print(np.max(abs(dldu-grad_U)))
print(np.max(abs(dldc-grad_c)))