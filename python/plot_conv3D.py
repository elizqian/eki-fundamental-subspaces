import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from linearEKI import *

plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('darkgrey')
    ax.spines['left'].set_color('darkgrey')
    ax.tick_params(axis='both', which='both', colors='darkgrey')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')

# construct a random 3x3 H with rank 2
basis,_ = np.linalg.qr(np.random.rand(3,3))
H = np.hstack((basis[:,:2],np.sum(basis[:,:2],axis=1)[:,np.newaxis])).T

# construct a random rank 2 ensemble with one component in range of H and one not in ran(H)
J = 15
v0 = basis[:,[0, 2]] @ np.random.rand(2,J)

prob = leastsquares(H=H)

# prob = leastsquares()
maxiter = 1000
det = EKI(prob,"det",maxiter,v0 = v0)
stoch = EKI(prob,"stoch",maxiter,v0 = det.v0)

orange = "#FFB320"
blue   = "#5BA9EF"
colors = [orange,blue]
styles = ["solid","dashed","dotted"]

cols = [det, stoch]
rows = ["misfit","error"]
projs = [["calPr","calQr","calNr"], ["bbPr","bbQr","bbNr"]]
lbls  = [["$\\|\\mathcal{P}_r\\theta_i^{(j)}\\|$","$\\|\\mathcal{Q}_r\\theta_i^{(j)}\\|$","$\\|\\mathcal{N}_r\\theta_i^{(j)}\\|$"],["$\\|\\mathbb{P}_r\\omega_i^{(j)}\\|$","$\\|\\mathbb{Q}_r\\omega_i^{(j)}\\|$","$\\|\\mathbb{N}_r\\omega_i^{(j)}\\|$"]]

lines = [[],[]]

fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5,3.5))

x = np.arange(maxiter+1)
xx = np.linspace(1.0001, maxiter+1)

for i in range(2): # row
    for j in range(2): # column
        for k in range(3): # linetype
            y = cols[j].getComponentNorm(rows[i],projs[i][k])
            if k == 0:
                scl = np.max(y[1,:])
            ln = axs[i,j].loglog(x,y,alpha=0.3,color=colors[i],linestyle=styles[k],label=lbls[i][k])

            if j == 0:
                lines[i].append(ln[0])
        sqrt = axs[i,j].loglog(xx,scl/np.sqrt(xx),color="grey",alpha=0.8)
        style_axes(axs[i,j])
        # axs.set_xlim()

axs[1,0].set_xlabel("Iteration number $i$")
axs[1,1].set_xlabel("Iteration number $i$")
axs[0,0].set_ylabel("Measurement space misfit")
axs[1,0].set_ylabel("State space error")
axs[0,0].set_title("Deterministic EKI")
axs[0,1].set_title("Stochastic EKI")

# legend for measurement space row
axs[0,1].legend(lines[0],lbls[0],loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# legend for state space row
axs[1, 1].legend(lines[1],lbls[1],loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# Legend for a 1/sqrt(i) rate line
proxy_line = Line2D([0], [0], color='gray', alpha=0.8, label="$1/\\sqrt{i}$ rate")
fig.legend(handles=[proxy_line], loc='center left', bbox_to_anchor=(0.75, 0.5), frameon=False)

plt.tight_layout()
fig.savefig("convrates_mpl.pdf")
