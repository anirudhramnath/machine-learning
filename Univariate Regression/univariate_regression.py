#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


def pad_ones(X, size):
	ones = np.ones((size, 1))
	X = np.concatenate((ones, X), axis=1)

	return X


def gradient_descent(X, y):
	m = y.size
	X = pad_ones(X, m)

	# init theta_0 and theta_1 to 0 
	theta = np.zeros((2, 1))
	iterations = 1500
	alpha = .01

	J_history = np.zeros((iterations, 1))

	for i in xrange(iterations):
		predictions = np.dot(X, theta)

		theta[0][0] = theta[0][0] -  alpha * (1.0/m) * (np.sum((predictions - y) * X[:,:1]) )
		theta[1][0] = theta[1][0] -  alpha * (1.0/m) * (np.sum((predictions - y) * X[:,1:]) )

		J_history[i][0] = compute_cost(X, y, theta)

	return theta, J_history
	

def compute_cost(X, y, theta):
	J, m = 0, y.size

	# Dot product of X and theta
	predictions = np.dot(X, theta)
	squared_error = (predictions - y) **2	
	J = np.sum(squared_error) / (2*m)

	return J


def load_data():
	data = np.loadtxt('univariate_regression_data.txt', delimiter=',')

	# do transpose since X and y are column vectors
	X, y = data[:,0][:,np.newaxis], data[:,1][:,np.newaxis]

	theta, J_histroy = gradient_descent(X, y)
	
	predict = lambda M, theta: pad_ones(M, M.size).dot(theta)
	
	plt.scatter(X, y, color='orange', label='data points')
	plt.plot()
	
	plt.xlabel('Population of City in 10,000s')
	plt.ylabel('Profit in 10,000s')

	M = np.linspace(5., 22., 100)[:, np.newaxis]

	plt.plot(M, predict(M, theta), color='black', label='linear regression')
	plt.legend()
	plt.show()

	X = np.arange(1500)
	
	plt.plot(X, J_histroy.flatten(), color='red', label='cost function')
	plt.legend()
	plt.show()

	print 'Results:\ntheta_0 = %.2f, theta_1 = %.2f' % (theta[0,0], theta[1,0]) 
	

if __name__ == '__main__':
	load_data()