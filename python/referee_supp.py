import numpy as np
from linearEKI import *
import pickle
import time


np.random.seed(2)

n = 500
d = 1000
J = 1000

start = time.time()

print("Setting up LS problem and EKI initial ensemble...")
prob,v0 = setupEKI(n,d,J)
v0small = v0[:,:100]
print("Setup complete: "+str(time.time()-start))

maxiter = 10000

# run deterministic and stochastic EKI for small and large ensembles, save results needed for plotting
det = EKI(prob,"det",maxiter,v0 = v0)

detsmall = EKI(prob,"det",maxiter,v0 = v0small)

stochsmall = EKI(prob,"stoch",maxiter,v0 = v0small)

stochlarge = EKI(prob,"stoch",maxiter,v0=v0)

with open("moreEnsembleSizes.pkl","wb") as f:
    pickle.dump([det,detsmall,stochsmall,stochlarge],f)
