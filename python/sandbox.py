# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.io as pio
from util_plots import *
from linearEKI import *

def normalize(v):
    return v/np.linalg.norm(v)

###########################################################################
# CODE TO SET UP AND RUN EKI
###########################################################################
# SETUP LEAST SQUARE PROBLEM
# construct a random 3x3 H with rank 2
basis,_ = np.linalg.qr(np.random.rand(3,3))
h1 = normalize(np.array([1,2,1])[:,np.newaxis])
h2 = normalize(np.array([2,1,1])[:,np.newaxis])
H = np.hstack((h1,h2,np.zeros((3,1))))

q,_ = np.linalg.qr(H.T)
q1 = q[:,0][:,np.newaxis]
q2 = q[:,1][:,np.newaxis]
q3 = q[:,2][:,np.newaxis]

meas = 0.8*np.array([-0.5,-0.2,0.3])[:,np.newaxis]
Sigma = np.array([[0.3, 0.1, 0.005],[0.1, 0.2, 0.003],[0.005,0.003,0.4]])

prob = leastsquares(H=H,meas = meas, Sigma = Sigma)

# SETUP EKI ITERATION
# construct a random rank 2 ensemble with one component in range of H and one not in ran(H)
J = 15
th = 2*np.pi*np.random.rand(1,J)
ps = np.pi*np.random.rand(1,J)
v0 = np.cos(th)*q1 + np.sin(th)*q3 + np.sin(ps)*q2

maxiter = 100
det = EKI(prob,"det",maxiter,v0 = v0)
stoch = EKI(prob,"stoch",maxiter,v0 = det.v0)

###########################################################################
# Plotting
###########################################################################

det.plot3Dhtml("deterministic3d.html")
stoch.plot3Dhtml("stochastic3D.html")
