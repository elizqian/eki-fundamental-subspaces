# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly.io as pio
from util_plots import *
from linearEKI import *

# def normalize(v):
#     return v/np.linalg.norm(v)

###########################################################################
# CODE TO SET UP AND RUN EKI
###########################################################################

J = 8
prob,v0 = setupNamedEKI("paperMatch",J)

maxiter = 100
det = EKI(prob,"det",maxiter,v0 = v0)
stoch = EKI(prob,"stoch",maxiter,v0 = det.v0)

###########################################################################
# Plotting
###########################################################################

det.plot3Dhtml("deterministic3d.html","<b>Deterministic EKI</b>")
stoch.plot3Dhtml("stochastic3D.html","<b>Stochastic EKI</b>")

# stoch.plot3Dhtml("temp.html","<b>Test</b>")