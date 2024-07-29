import numpy as np
from util_plots import *


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
            # Sigma = np.array([[1,0.1,0.001],[0.1,4,0.05],[0.001,0.05,16]])
        self.Sigma = Sigma

        # set up measurement, min norm pseudoinverse, min norm solution
        if meas is None:
            truth = np.random.rand(d,1)
            meas  = H @ truth + np.random.multivariate_normal(np.zeros((n,)),Sigma)[:,np.newaxis]
        self.meas = meas
        self.Hplus = np.linalg.pinv(H.T @ np.linalg.solve(Sigma, H)) @ (np.linalg.solve(Sigma,H).T)
        self.vstar = self.Hplus @ meas

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
            m = ls.meas + np.random.multivariate_normal(np.zeros((ls.n,)),ls.Sigma)[:,np.newaxis]
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

        # normalize first r in-space vectors
        r = np.sum(np.abs(delta)>1e-10)
        for i in range(r):
            W[:,i] = W[:,i]/np.sqrt(W[:,i].T @ Sigma @ W[:,i])
        
        # compute basis for Ran(Sigma^-1 * H)
        q1,_,_ = np.linalg.svd(np.linalg.solve(Sigma,H))
        q1 = q1[:,:h]

        # compute basis for kernel of H
        q2,_,_ = np.linalg.svd(H.T)
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

        self.calPr = Sigma @ W[:,:r] @ W[:,:r].T 
        if h > r:
            self.calQr = Sigma @ W[:,r:h] @ W[:,r:h].T 
        else:
            self.calQr = np.zeros((n,n))
        self.calNr = np.eye(n) - self.calPr - self.calQr 

        U = np.zeros((d,d))
        for ell in range(h):
            if ell < r:
                U[:,ell] = Gamma @ H.T @ W[:,ell]/delta[ell]
            else:
                U[:,ell] = Hplus @ Sigma @ W[:,ell]
        
        self.bbPr = U[:,:r] @ U[:,:r].T @ fisher
        if h > r:
            self.bbQr = U[:,r:h] @ U[:,r:h].T @ fisher 
        else:
            self.bbQr = np.zeros((d,d))
        self.bbNr = np.eye(d) - self.bbPr - self.bbQr

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
    
    # def get3DPlotData(self):
    #     # this function is only defined for specific dimensions

    #     assert np.linalg.matrix_rank(self.ls.H)==2
    #     Gam = np.cov(self.v0)
    #     assert np.linalg.matrix_rank(Gam)==2
    #     assert np.linalg.matrix_rank(self.bbPr)==1

    #     names = ["Pr","Qr","Nr"]
    #     basisM = np.zeros((3,3))
    #     basisS = np.zeros((3,3))
    #     for j in range(3):
    #         proj = self.__getattribute__("bb"+names[j])
    #         temp = proj @ np.random.rand(3)
    #         basisS[:,j] = temp / np.linalg.norm(temp)

    #         proj = self.__getattribute__("cal"+names[j])
    #         temp = proj @ np.random.rand(3)
    #         basisM[:,j] = temp / np.linalg.norm(temp)
        
    #     q,_,_ = np.linalg.svd(self.ls.H)
    #     qH  = q[:,:2]
    #     q,_,_ = np.linalg.svd(self.ls.H.T)
    #     qHT = q[:,:2]
        
    #     q,_,_ = np.linalg.svd(Gam)
    #     qGam = q[:,:2]
    #     return basisM,basisS,qH,qHT,qGam
    
    def plotSubspaces(self,fig,space):
        if space == "state":
            prefix = "bb"
            c = 1
        elif space == "measurement":
            prefix = "cal"
            c = 2
        names = ["Pr","Qr","Nr"]
        colors = [cobalt,persimmon,maroon]
        for i in range(3):
            proj = self.__getattribute__(prefix+names[i])
            v,sig,_ = np.linalg.svd(proj)
            dim = sum(sig>1e-8)
            v = np.squeeze(v[:,:dim])
            if dim == 1:
                plot_vector3(fig,1,c,v,"Ran("+names[i]+")",colors[i])
            elif dim == 2:
                plot_plane(fig,1,c,v[:,0],v[:,1],"Ran("+names[i]+")",grid_size = 1, color=colors[i])
            elif dim == 3:
                pass
            elif dim == 0:
                pass
            # if i == 1:
            #     print(v)
        


    def plot3Dhtml(self,savename):

        # we only generate 3D plots for 3D problems
        assert self.ls.d == 3
        assert self.ls.n == 3

        # get basis vectors for plotting

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=("State Space", "Measurement Space"),
            horizontal_spacing = 0.1
        )

        # plot range of H and H^T
        qH,sigH,qHT = np.linalg.svd(self.ls.H)
        h = sum(sigH>1e-8)
        qH  = qH[:,:h]
        qHT = qHT[:,:h]
        if h == 2: # plot planes
            plot_plane(fig,1,1,qHT[:,0], qHT[:,1], "Ran(H^T)",grid_size=1, color = orange)
            plot_plane(fig,1,2,qH[:,0],qH[:,1], "Ran(H)",grid_size=1, color = orange)
        elif h == 1: # plot vectors
            plot_vector3(fig,1,1,qHT,"Ran(H^T)",color=orange)
            plot_vector3(fig,1,2,qH,"Ran(H)",color=orange)
        
        # plot range of Gamma_i
        Gam = np.cov(self.v0)
        qG,sigG,_ = np.linalg.svd(Gam)
        g = sum(sigG>1e-8)
        qG = qG[:,:g]
        if g == 2:
            plot_plane(fig,1,1,qG[:,0],qG[:,1],"Ran(Gamma_i)",grid_size=1,color=blue)
        elif g == 1:
            plot_vector3(fig,1,1,qG,"Ran(Gamma_i)",color=blue)
        
        # left subplot (state space) -- plot particles, vstar, subspaces
        plot_paths(fig,1,1,self.vv,blue,"State particle paths")
        plot_point3(fig,1,1,np.squeeze(self.ls.vstar), "v*", "black",sym="cross")
        self.plotSubspaces(fig,"state")

        # right subplot (measurement space) -- plot particles, vstar, subspaces
        hh = self.ls.H[np.newaxis,:,:] @ self.vv
        plot_paths(fig,1,2,hh,orange,"Measurement particle paths")
        plot_point3(fig,1,2,np.squeeze(self.ls.meas), "m", "black",sym="cross")
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
                    xaxis_title='x1',
                    yaxis_title='x2',
                    zaxis_title='x3',
                    aspectmode='cube'
                ),
                scene2=dict(
                    xaxis_title='y1',
                    yaxis_title='y2',
                    zaxis_title='y3',
                    aspectmode='cube'
                ),
            )

        pio.write_html(fig, file=savename, auto_open=True)

