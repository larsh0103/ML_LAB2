import numpy as np
import random
import math 
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def k_linear(xi,xj):
	return np.transpose(xi).dot(xj)

def k_quadratic(xi,xj):
	return (np.transpose(xi).dot((xj))+1)**2



N = 20

classA = np.concatenate((np.random.randn(N/2,2)*0.2 + [1.5,0.5], 
	np.random.randn(N/2,2)*0.2 +[-1.5,0.5]))
classB = np.random.randn(N,2)*0.2 + [0,-0.5]

inputs = np.concatenate((classA,classB))
print("shape inputs", inputs.shape)
targets = np.concatenate((np.ones(classA.shape[0]),np.ones(classB.shape[0])*-1))

permute = list(range(inputs.shape[0]))
random.shuffle(permute)
inputs =inputs[permute, :]
targets = targets[permute]


P = np.array( [[targets[i]*targets[j]*k_quadratic(inputs[i],inputs[j]) for i in range(len(inputs))] 
	for j in range(len(inputs))])


def indicator(y,x,alpha,beta):
	s=([x,y])
	return np.sum((alpha*targets).dot([k_quadratic(s,inputs[i]) for i in range(len(inputs))])) - beta

def plot_data(classA,classB):
	plt.plot([p[0] for p in classA], [p[1] for p in classA],'b.')
	plt.plot([p[0] for p in classB], [p[1]  for p in classB],'r.')
	plt.axis('equal')
	plt.show()

def plot_result(classA,classB,answer):
	plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
	plt.plot([p[0] for p in classB],[p[1] for p in classB],'r.')
	plt.plot([p[0] for p in answer], 'g.')
	plt.show()

def objective(alpha):
	return 1/2* np.sum(P.dot(alpha) - np.sum(alpha))

def zero_fun(alpha):
	print(sum(alpha[i] * targets[i] for i in range(2*N)))
	return (sum(alpha[i] * targets[i] for i in range(2*N)))

def beta(s):
	return (sum(alpha[i]*targets[i]*k_quadratic(s[1:3],inputs[i]) for i in range(2*N))) -s[3]

if __name__ == '__main__':
	print(P.shape)
	print("targets", targets)
	print("inputs", inputs)
	print("P", P)
	plot_data(classA,classB)
	C = 10
	start = np.random.rand(2*N)
	print("start", start)
	constraint = {'type':'eq','fun':zero_fun}
	ret = minimize(objective, start, bounds = [(0,C) for b in range(2*N)], constraints=constraint)
	alpha =ret['x']
	print("zero sum of result", zero_fun(alpha))
	answer=np.zeros(shape=(2*N,4))
	for i in range(len(alpha)):
		if alpha[i]>10**(-5):
			answer[i][0]=alpha[i]
			answer[i][1:3]=inputs[i]
			answer[i][3]=targets[i]
	print(ret)
	print(answer[:,0])
	non_zero= [ i  for i in range(2*N) if answer[i][0] !=0] 
	beta = beta(answer[non_zero[2]])
	plot_result(classA,classB,answer)
	xgrid = np.linspace(-5,5)
	ygrid = np.linspace(-4,4)
	grid = np.array([[indicator(x,y,answer[:,0],beta) for y in ygrid] for x in xgrid])
	plt.contour(xgrid,ygrid,grid,(-1.0,0.0,1,0), 
		colors=('red','black','blue'),linewidths=(1,3,1))

