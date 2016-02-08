#hwk 2, Nathan Singer
'''
    
'''

import numpy as np
from scipy.linalg import inv
from numpy import linalg as la
from matplotlib import pyplot as plt

def compute_trajectory(which_compute = False):
    
    b = 10E-4
    g = 9.8
    t = .1 #this is the delta t
    sx,sy,vx,vy = 0,0,300,600
    Q = .1*np.eye(4)
    R = 500*np.eye(2)
    F = np.array(([1.,0.,t,0.],[0.,1.,0.,t],[0.,0.,1-b,0.],[0.,0.,0.,1-b]))
    u = np.array([0,0,0,-g*t])
    x0 = np.array([sx,sy,vx,vy])
    rand_vec = np.random.rand(4)
    w = np.dot((la.cholesky(Q)).T, rand_vec)
    N = 1200
    x_list = []

    if which_compute == True:
        y = []
        H = np.eye(4)[:2]
        R = np.eye(2)*500

    for i in xrange(N):
        if which_compute == True and i+1 >= 400 and i+1 <= 600:
            y.append(np.dot(H,x0) + np.dot(la.cholesky(R),np.random.randn(2)))
        x_new = np.dot(F,x0) + u + np.dot(la.cholesky(Q),np.random.randn(4))
        x0 = x_new
        x_list.append(x_new)


    if which_compute == True:
        return np.array(x_list), np.array(y) #.shape
    else:
        return np.array(x_list)


def prob1():
    
    print compute_trajectory()[-1] #in this case which_compute is false

'''
   output = [ 21857.45201493  -7772.87114786    100.19284715   -508.31841942]
'''


def prob2():
    x,y = compute_trajectory(which_compute = True) #we set which_compute to true to compute x and y
    plt.plot(x[400:600,0],x[400:600,1])
    plt.scatter(y[:,0],y[:,1])
    plt.title("Problem 2")
    plt.show()

def kalman(P,x,y,F,G,Q,R,U,H):
    def step(P,x,y,F,G,Q,R,U,H):
        nextP = inv(inv(Q+F.dot(P).dot(F.T)) + H.T.dot(inv(R)).dot(H))
        m1 = F.dot(x)
        m2 = U
        m31 = nextP.dot(H.T).dot(inv(R))
        m321 = H.dot(F.dot(x) + U)
        m32 = m321 - y
        nextx = m1+m2-m3
        return list(nextP), list(nextx)

    P = [10**6*Q]
    x = [[0,0,0,0]]
    steps = 200
    for k in xrange(steps - 1):
        nextP, nextx = step(P[k],x[k],y[k],F,G,Q,R,U,H)
        P.append(nextP)
        x.append(nextx)
    return np.array(P), np.array(x)

def prob3():
    b = 10E-4
    g = 9.8
    G = 1.
    t = .1 #this is the delta t
    sx,sy,vx,vy = 0,0,300,600
    Q = .1*np.eye(4)
    R = 500*np.eye(2)
    F = np.array(([1.,0.,t,0.],[0.,1.,0.,t],[0.,0.,1-b,0.],[0.,0.,0.,1-b]))
    u = np.array([0,0,0,-g*t])
    P0 = 10**6*Q
    H = np.eye(4)[:2]
    y = compute_trajectory(which_compute = True)[1]
    print y
    y = y[400:600]
    YY = kalman(P0,y,F,G,Q,R,u,H)[1]
    plt.plot(YY[1:,0],YY[1:,1])
    plt.scatter(y[:,0],y[:,1])
    plt.show()

def prob4():
    pass

def prob5():
    b = 10E-4
    t = .1
    g = 9.8
    F = np.array(([1.,0.,t,0.],[0.,1.,0.,t],[0.,0.,1-b,0.],[0.,0.,0.,1-b]))
    F_inv = inv(F)
    u = np.array([0,0,0,-g*t])
    
    Q = .1*np.eye(4)

    x_old = np.array([0,0,300,600])
    x = []
    N = 1200

    for i in xrange(N):
        x_new = np.dot(F_inv,x_old) - np.dot(F_inv,u) - np.dot(F_inv,np.dot(la.cholesky(Q),np.random.normal(size=4)))
        x_old = x_new
        x.append(x_new)

    print np.array(x)

'''
output = [[ -2.96359487e+01  -6.01092124e+01   3.00681246e+02   6.01588918e+02]
 [ -5.98382521e+01  -1.19966280e+02   3.01540822e+02   6.02309996e+02]
 [ -8.99772099e+01  -1.79952364e+02   3.01538644e+02   6.03663226e+02]
 ..., 
 [ -7.09427860e+04  -2.49227019e+05   1.01867475e+03   4.26898328e+03]
 [ -7.10446567e+04  -2.49654144e+05   1.01989717e+03   4.27446470e+03]
 [ -7.11467681e+04  -2.50082365e+05   1.02099361e+03   4.27973770e+03]]
'''

#print prob1()
#print prob2()
prob3()
#print prob5()
#print np.random.random(size = 4)
