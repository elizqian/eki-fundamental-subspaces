import numpy as np
from util_plots import *

def normalize(v):
    return v/np.linalg.norm(v)

class leastsquares:
    def __init__(self,H=None, Sigma = None, meas = None):
        # setup observation operator H and problem dimensions
        if H is None:
            H = np.array( [[1, 0, 0],[0, 1, 0],[0, 0,  0]])
        n,d = H.shape
        self.H = H
        self.n = n
        self.d = d

        # setup observation noise
        if Sigma is None:
            Lsig = 0.5*np.random.rand(n,n)
            Sigma = Lsig @ Lsig.T
        self.Sigma = Sigma

        # set up measurement, min norm pseudoinverse, min norm solution
        if meas is None:
            truth = np.random.rand(d,1)
            meas  = H @ truth + np.random.multivariate_normal(np.zeros((n,)),Sigma)[:,np.newaxis]
        self.meas = meas
        self.Hplus = np.linalg.pinv(H.T @ np.linalg.solve(Sigma, H)) @ (np.linalg.solve(Sigma,H).T)
        self.vstar = self.Hplus @ meas

def setupNamedEKI(name,J):
    if name == "random3D":
        # n=d=3 case where each fundamental subspace is 1D, and the subspaces are randomly generated
        basis,_ = np.linalg.qr(np.random.rand(3,3))
        H = np.hstack((basis[:,:2],np.sum(basis[:,:2],axis=1)[:,np.newaxis])).T
        v0 = basis[:,[0, 2]] @ np.random.rand(2,J)
        ls = leastsquares(H=H)
    elif name == "illustrate3D1D":
        # n=d=3 case where each fundamental subspace is 1D and things are set up so that the plots are nice-ish
        basis,_ = np.linalg.qr(np.random.rand(3,3))
        h1 = normalize(np.array([1,2,1])[:,np.newaxis])
        h2 = normalize(np.array([2,1,1])[:,np.newaxis])
        H = np.hstack((h1,h2,np.zeros((3,1))))

        q,_,_ = np.linalg.svd(H.T)
        q3 = q[:,2][:,np.newaxis]

        meas = 0.8*np.array([-0.5,-0.2,0.3])[:,np.newaxis]
        Sigma = np.array([[0.3, 0.1, 0.005],[0.1, 0.2, 0.003],[0.005,0.003,0.4]])

        ls = leastsquares(H=H,meas = meas, Sigma = Sigma)
        th = 2*np.pi*np.random.rand(1,J)
        v0 = np.cos(th)*h1 + np.sin(th)*q3 

    elif name == "illustrate3D2D":
        # n=d=3 case where ran(Pr) is 2d, ran(Qr)=0, and ran(Nr) is 1d and things are set up for nice-ish plots
        h1 = normalize(np.array([1,2,1])[:,np.newaxis])
        h2 = normalize(np.array([2,1,1])[:,np.newaxis])
        H = np.hstack((h1,h2,np.zeros((3,1))))

        q,_ = np.linalg.qr(H.T)
        q1 = q[:,0][:,np.newaxis]
        q2 = q[:,1][:,np.newaxis]
        q3 = q[:,2][:,np.newaxis]

        meas = 0.8*np.array([-0.5,-0.2,0.3])[:,np.newaxis]
        Sigma = np.array([[0.3, 0.1, 0.005],[0.1, 0.2, 0.003],[0.005,0.003,0.4]])
        ls = leastsquares(H=H,meas = meas, Sigma = Sigma)

        th = 2*np.pi*np.random.rand(1,J)
        ps = np.pi*np.random.rand(1,J)
        v0 = np.cos(th)*q1 + np.sin(th)*q3 + np.sin(ps)*q2

    elif name == "paperMatch":
        n = 3
        d = 3
        H = np.array( [[1, 0, 0],[0, 1, 0],[0, 0,  0]])
        v1 = np.array([0,1,0])
        v2 = np.array([0,0,1])
        
        th = 2*np.pi*np.random.rand(J,1)
        v0 = (np.cos(th) * v1 + np.sin(th) * v2).T

        meas = np.array([0.75, 0.25, 0.5])[:,np.newaxis]
        # Sigma = np.array([[0.45951155, 0.22958299, 0.32095932],       [0.22958299, 0.27617495, 0.2561246 ],       [0.32095932, 0.2561246 , 0.42785381]])
        ls = leastsquares(H=H,meas=meas)

    return ls,v0

def setupEKI(n,d,J):
    H = np.random.rand(n,d)
    v = np.random.rand(d,1)
    v = normalize(v)
    H = H - H @ (v @ v.T)

    w = np.random.rand(n,1)
    w = normalize(w)
    H = (H.T - H.T @ (w @ w.T)).T

    ls = leastsquares(H=H)
    u,_,_ = np.linalg.svd(H.T)
    basisV = np.hstack((v,u[:,:d-2]))

    v0 = basisV @ np.random.rand(d-1,J) 
    q = H.T[:,0][:,np.newaxis]
    q = normalize(q)
    v0 = v0 - (q @ q.T @ v0)
    return ls,v0

class EKI:
    def __init__(self,lsprob,opt,maxiter,v0=None):

        # initialize ensemble from input or use default case
        if v0 is None:
            J = 15
            d = 3
            v0 = np.vstack((np.random.rand(J),0.1*np.ones(J,),0.5*np.random.rand(J)))
        else:
            d,J = v0.shape
        self.v0 = v0.copy()
        self.v = v0
        self.J = J

        # add LS problem details to object
        self.ls = lsprob

        # initialize storage for iteration and iteration counter
        vv = np.zeros((maxiter+1,d,J))
        vv[0,:,:] = v0

        # run iteration
        for i in range(maxiter):
            self.update(opt)
            vv[i+1,:,:] = self.v
        
        self.vv = vv

        self.specdecomp()

    def update(self,opt):
        ls = self.ls
        Gamma = np.cov(self.v)
        S = (ls.H @ Gamma @ ls.H.T + ls.Sigma)
        K = (np.linalg.solve(S,ls.H @ Gamma)).T
        if opt == "det":
            m = ls.meas
        elif opt == "stoch":
            eps = np.random.multivariate_normal(np.zeros((ls.n,)),ls.Sigma,size=(self.J)).T
            m = ls.meas + eps
        self.v = (np.eye(ls.d) - K @ ls.H) @ self.v + K @ m
    
    def specdecomp(self):
        H     = self.ls.H
        Sigma = self.ls.Sigma
        v     = np.squeeze(self.v0)
        n,d   = H.shape
        h = np.linalg.matrix_rank(H)

        fisher = H.T @ np.linalg.solve(Sigma,H)
        Hplus = self.ls.Hplus
        Gamma = np.cov(v)
        HGamH = H @ Gamma @ H.T

        ###########################################################
        # calculate, re-order, and re-normalize eigenvalues/vectors
        ###########################################################
        delta,W = np.linalg.eig(np.linalg.solve(Sigma,HGamH))
        idx = delta.argsort()[::-1]
        delta = delta[idx]
        W     = W[:,idx]

        self.delta = delta

        # normalize first r in-space vectors
        r = np.sum(np.abs(delta)>1e-6)
        self.delta = delta
        for i in range(r):
            W[:,i] = W[:,i]/np.sqrt(W[:,i].T @ Sigma @ W[:,i])
        
        # compute basis for Ran(Sigma^-1 * H)
        q1,_,_ = np.linalg.svd(np.linalg.solve(Sigma,H))
        q1 = q1[:,:h]

        # compute basis for kernel of H.T
        q2,_,_ = np.linalg.svd(H)
        q2 = q2[:,h:]
        
        for ell in range(r,n):
            temp = np.random.rand(n,)
            if ell < h:
                w = q1 @ q1.T @ temp 
            else:
                w = q2 @ q2.T @ temp 
            for k in range(ell):
                w = w - ((w.T @ Sigma @ W[:,k]) * W[:,k])
            W[:,ell] = w/np.sqrt(w.T @ Sigma @ w)
        
        self.W = W

        self.calP = Sigma @ W[:,:r] @ W[:,:r].T 
        if h > r:
            self.calQ = Sigma @ W[:,r:h] @ W[:,r:h].T 
        else:
            self.calQ = np.zeros((n,n))
        self.calN = np.eye(n) - self.calP - self.calQ 

        U = np.zeros((d,d))
        for ell in range(h):
            if ell < r:
                U[:,ell] = Gamma @ H.T @ W[:,ell]/delta[ell]
            else:
                U[:,ell] = Hplus @ Sigma @ W[:,ell]
        
        self.bbP = U[:,:r] @ U[:,:r].T @ fisher
        if h > r:
            self.bbQ = U[:,r:h] @ U[:,r:h].T @ fisher 
        else:
            self.bbQ = np.zeros((d,d))
        self.bbN = np.eye(d) - self.bbP - self.bbQ

    def getComponentNorm(self,qoi,projName):
        if qoi == "state":
            q = self.vv
        elif qoi == "meas":
            q = self.ls.H[np.newaxis,:,:] @ self.vv 
        elif qoi == "error":
            q = self.vv - self.ls.vstar
        elif qoi == "misfit":
            q = self.ls.H[np.newaxis,:,:] @ self.vv - self.ls.meas
        
        proj = self.__getattribute__(projName)
        comp = proj[np.newaxis,:,:] @ q 
        return np.sqrt(np.sum((comp**2),axis=1))
    
    def plotSubspaces(self,fig,space):
        if space == "state":
            prefix = "bb"
            c = 2
        elif space == "measurement":
            prefix = "cal"
            c = 1
        names = ["P","Q","N"]
        colors = [darkblue, orange, black]
        for i in range(3):
            proj = self.__getattribute__(prefix+names[i])
            v,sig,_ = np.linalg.svd(proj)
            dim = sum(sig>1e-8)
            v = np.squeeze(v[:,:dim])
            if v[i] < 0:
                v = -v
            if dim == 1:
                plot_vector3(fig,1,c,v,"Ran("+names[i]+")",colors[i])
            elif dim == 2:
                plot_plane(fig,1,c,v[:,0],v[:,1],"Ran("+names[i]+")",grid_size = 1, color=colors[i])
            elif dim == 3:
                pass
            elif dim == 0:
                pass

    def plot3Dhtml(self,savename,title):

        # we only generate 3D plots for 3D problems
        assert self.ls.d == 3
        assert self.ls.n == 3

        # get basis vectors for plotting
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=("Observation Space", "State Space"),
            horizontal_spacing = 0.1
        )

        # plot range of H and H^T
        qH,sigH,qHT = np.linalg.svd(self.ls.H)
        h = sum(sigH>1e-8)
        qH  = qH[:,:h]
        qHT = qHT[:,:h]
        if h == 2: # plot planes
            plot_plane(fig,1,2,qHT[:,0], qHT[:,1], "Ran(H^T)",grid_size=1, color = orange)
            plot_plane(fig,1,1,qH[:,0],qH[:,1], "Ran(H)",grid_size=1, color = orange)
        elif h == 1: # plot vectors
            plot_vector3(fig,1,2,qHT,"Ran(H^T)",color=orange)
            plot_vector3(fig,1,1,qH,"Ran(H)",color=orange)
        
        # plot range of Gamma_i
        Gam = np.cov(self.v0)
        qG,sigG,_ = np.linalg.svd(Gam)
        g = sum(sigG>1e-8)
        qG = qG[:,:g]
        if g == 2:
            plot_plane(fig,1,2,qG[:,0],qG[:,1],"Ran(Gamma_i)",grid_size=1,color=blue)
        elif g == 1:
            plot_vector3(fig,1,2,qG,"Ran(Gamma_i)",color=blue)
        
        vEKI = np.mean(self.vv[-1,:,:],axis=1)
        # right subplot (state space) -- plot particles, vstar, subspaces
        plot_paths(fig,1,2,self.vv,blue,"State particle paths")
        plot_star3(fig,1,2,np.squeeze(self.ls.vstar), "v*",orange,qHT[:,0],qHT[:,1])
        plot_star3(fig,1,2,np.squeeze(self.bbP @ self.ls.vstar), "Pv*",darkblue,qG[:,0],qG[:,1])
        
        self.plotSubspaces(fig,"state")

        # left subplot (measurement space) -- plot particles, vstar, subspaces
        hh = self.ls.H[np.newaxis,:,:] @ self.vv
        plot_paths(fig,1,1,hh,blue,"Observation particle paths")
        plot_point3(fig,1,1,np.squeeze(self.ls.meas), "y", black,sym="cross")
        plot_star3(fig,1,1,np.squeeze(self.ls.H @self.ls.vstar), "Hv*",orange,qH[:,0],qH[:,1])
        plot_star3(fig,1,1,np.squeeze(self.ls.H @ vEKI), "H(v_EKI)",darkblue,qH[:,0],qH[:,1])
        self.plotSubspaces(fig,"measurement")

        fig.update_layout(
                height=600,  
                showlegend=True,
                legend=dict(
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='middle',
                    orientation='v'
                ),
                scene=dict(
                    xaxis_title='y1',
                    yaxis_title='y2',
                    zaxis_title='y3',
                    aspectmode='cube'
                ),
                scene2=dict(
                    xaxis_title='v1',
                    yaxis_title='v2',
                    zaxis_title='v3',
                    aspectmode='cube'
                ),
                title={
                'text': title,
                'x': 0.5,  # Center the title
                'xanchor': 'center',  # Anchor the title at the center
                'yanchor': 'top'
            }
            )

        pio.write_html(fig, file=savename, auto_open=True)

