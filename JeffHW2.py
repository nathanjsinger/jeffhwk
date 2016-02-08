import numpy as np
from numpy import linalg as la
from scipy import sparse
from matplotlib import pyplot as plt

def projectile_model(x0=np.array([0,0,300,600]),t_change=.1,iters=1200,measure=False,measure_start=400,measure_stop=600,Q_scale=.1):
	b = 10E-4
	F = np.array(sparse.diags(np.array([[1,1,1-b,1-b],[t_change,t_change]]),[0,2]).todense())
	u = np.zeros(4)
	g = 9.8
	u[-1] -= g*t_change
	Q = np.eye(4)*Q_scale
	x_old = x0
	x = []
	if measure == True:
		y = []
		H = np.eye(4)[:2]
		R = np.eye(2)*500
	
	for i in xrange(iters):
		if measure == True and i+1 >= measure_start and i+1 <= measure_stop:
			y.append(np.dot(H,x_old) + np.dot(la.cholesky(R),np.random.normal(size=2)))
		x_new = np.dot(F,x_old) + u + np.dot(la.cholesky(Q),np.random.normal(size=4))
		x_old = x_new
		x.append(x_new)
	if measure == True:
		return np.array(x), np.array(y)
	else:
		return np.array(x)


def problem1():
	return projectile_model()[-1]
	
print problem1()

def problem2():
	x,y = projectile_model(iters=600,measure=True)
	plt.plot(x[400:600,0],x[400:600,1])
	plt.plot(y[:,0],y[:,1])
	plt.show()
	
#problem2()

def problem3():
	x,y = projectile_model(iters=400,measure=True,measure_start=390,measure_stop=400)
	y400 = y[-1]
	
	#x_velocity_estimate = (y[0][0] - y[-1][0])/10
	#y_velocity_estimate = (y[0][1] - y[-1][1])/10
	#x_velocity_estimate = np.average(x[0::40,0])
	#y_velocity_estimate = np.average(x[-10:,1])
	x_velocity_estimate = 60
	y_velocity_estimate = 30
	
	x0 = np.array([y400[0],y400[1],x_velocity_estimate,y_velocity_estimate])
	x3 = projectile_model(x0=x0,iters=200)
	
	x2,y2 = projectile_model(iters=600,measure=True)
	plt.plot(x2[400:600,0],x2[400:600,1])
	plt.plot(y2[:,0],y2[:,1])
	
	plt.plot(x3[:,0],x3[:,1])
	plt.show()
	
#problem3() #this doesn't work at all likes it's supposed to. Problem is not very clear.

def problem4():
	#can't do this problem without a correct problem 3
	pass

#backwards iteration
def problem5(x_end,t_change=.1,iters=1200,Q_scale=.1):
	b = 10E-4
	F = np.array(sparse.diags(np.array([[1,1,1-b,1-b],[t_change,t_change]]),[0,2]).todense())
	F_inv = la.inverse(F)
	u = np.zeros(4)
	g = 9.8
	u[-1] -= g*t_change
	Q = np.eye(4)*Q_scale
	x_old = x0
	x = []
	
	for i in xrange(iters):
		x_new = np.dot(F_inv,x_old) - np.dot(F_inv,u) - np.dot(F_inv,np.dot(la.cholesky(Q),np.random.normal(size=4)))
		x_old = x_new
		x.append(x_new)

	print np.array(x)

#print problem5