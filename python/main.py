import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from linearEKI import *

##############################
# set EKI example
##############################
np.random.seed(0)   # for reproducibility of plots in paper

n = 8   # number of observations
d = 12  # number of states
J = 40  # number of particles in the large ensemble

# this function sets up a random LS problem and EKI initial ensemble whose fundamental subspaces are all non-trivial
prob,v0 = setupEKI(n,d,J)   
v0small = v0[:,:5]      # use just the first 5 particles for small ensemble tests

##############################
# run EKI 
##############################
maxiter = 1000
det = EKI(prob,"det",maxiter,v0 = v0small)          # deterministic
stochsmall = EKI(prob,"stoch",maxiter,v0 = v0small) # stochastic small ensemble
stochlarge = EKI(prob,"stoch",maxiter,v0=v0)        # stoch large ensemble


##############################
# plotting setup and plot
##############################
plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts,amsmath}'

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('darkgrey')
    ax.spines['left'].set_color('darkgrey')
    ax.tick_params(axis='both', which='both', colors='darkgrey')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')

orange = "#E97132"
blue = "#6ABCEB"
black  = "#888888"
colors = [blue,orange,black]
styles = ["solid","dashed","dotted"]

cols = [det, stochlarge,stochsmall]
rows = ["misfit","error"]
projs = [["calP","calQ","calN"], ["bbP","bbQ","bbN"]]
lbls  = [["$\\|\\boldsymbol{\\mathcal{P}}\\boldsymbol{\\theta}_i^{(j)}\\|$","$\\|\\boldsymbol{\\mathcal{Q}}\\boldsymbol{\\theta}_i^{(j)}\\|$","$\\|\\boldsymbol{\\mathcal{N}}\\boldsymbol{\\theta}_i^{(j)}\\|$"],["$\\|\\mathbb{P}\\boldsymbol{\\omega}_i^{(j)}\\|$","$\\|\\mathbb{Q}\\boldsymbol{\\omega}_i^{(j)}\\|$","$\\|\\mathbb{N}\\boldsymbol{\\omega}_i^{(j)}\\|$"]]

lines = [[],[]]

fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(5.5,4))

x = np.arange(maxiter+1)
xx = np.linspace(1.0001, maxiter+1)

for i in range(2): # row
    for j in range(3): # column
        for k in [1,2,0]: # linetype
            y = cols[j].getComponentNorm(rows[i],projs[i][k])
            if k == 0:
                scl = np.max(y[1,:])
            ln = axs[i,j].loglog(x,y,alpha=0.3,color=colors[k],linestyle=styles[k],label=lbls[i][k])
            prox = Line2D([0], [0], color=colors[k],linestyle=styles[k], label=lbls[i][k])
            if j == 0:
                lines[i].append(prox)
        sqrt = axs[i,j].loglog(xx,scl/np.sqrt(xx),color="#555555",alpha=0.8)
        style_axes(axs[i,j])


axs[1,1].set_xlabel("Iteration number $i$",fontsize=12,labelpad=2)
axs[0,0].set_ylabel("Measurement space\n misfit",fontsize=12,labelpad=1)
axs[1,0].set_ylabel("State space\n residual",fontsize=12,labelpad=1)
axs[0,1].set_title("(Large ensemble)",pad=1)
axs[0,2].set_title("(Small ensemble)",pad=1)
fig.text(0.7, 0.95, 'Stochastic EKI', ha='center', fontsize=13)
fig.text(0.27, 0.95, 'Deterministic EKI', ha='center', fontsize=13)

# legend
lines_all = [lines[0][2],lines[1][2],lines[0][0],lines[1][0],lines[0][1],lines[1][1]]
lbls_all  = [lbls[0][0],lbls[1][0],lbls[0][1],lbls[1][1],lbls[0][2],lbls[1][2]]
axs[0,0].legend(lines_all,lbls_all,loc='lower left',bbox_to_anchor=(-0.1,-0.7),ncols=3,frameon=False,fontsize=12,handletextpad=0.2,columnspacing = 1)

# Legend for a 1/sqrt(i) rate line
proxy_line = Line2D([0], [0], color='#555555', alpha=0.8, label="$1/\\sqrt{i}$ rate")
fig.legend(handles=[proxy_line], loc='center left', bbox_to_anchor=(0.75, 0.485), frameon=False,handletextpad=0.2,fontsize=12)

plt.subplots_adjust(top=0.88,right=0.98,left=0.13,bottom=0.1,hspace=0.7,wspace=0.1)
fig.savefig("EKIconvergence.pdf")
plt.close()
