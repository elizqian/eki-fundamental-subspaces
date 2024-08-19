import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import loadmat
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

n = 8
d = 12

J = 40


prob,v0 = setupEKI(n,d,J)
v0small = v0[:,:5]

maxiter = 1000
det = EKI(prob,"det",maxiter,v0 = v0small)
stochsmall = EKI(prob,"stoch",maxiter,v0 = v0small)
stochlarge = EKI(prob,"stoch",maxiter,v0=v0)

orange = "#FFB320"
blue   = "#5BA9EF"
colors = [orange,blue]
styles = ["solid","dashed","dotted"]

cols = [det, stochlarge,stochsmall]
rows = ["misfit","error"]
projs = [["calPr","calQr","calNr"], ["bbPr","bbQr","bbNr"]]
lbls  = [["$\\|\\mathcal{P}\\theta_i^{(j)}\\|$","$\\|\\mathcal{Q}\\theta_i^{(j)}\\|$","$\\|\\mathcal{N}\\theta_i^{(j)}\\|$"],["$\\|\\mathbb{P}\\omega_i^{(j)}\\|$","$\\|\\mathbb{Q}\\omega_i^{(j)}\\|$","$\\|\\mathbb{N}\\omega_i^{(j)}\\|$"]]

lines = [[],[]]

fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(6.5,3.5))

x = np.arange(maxiter+1)
xx = np.linspace(1.0001, maxiter+1)

for i in range(2): # row
    for j in range(3): # column
        for k in range(3): # linetype
            y = cols[j].getComponentNorm(rows[i],projs[i][k])
            if k == 0:
                scl = np.max(y[1,:])
            ln = axs[i,j].loglog(x,y,alpha=0.3,color=colors[i],linestyle=styles[k],label=lbls[i][k])

            if j == 0:
                lines[i].append(ln[0])
        sqrt = axs[i,j].loglog(xx,scl/np.sqrt(xx),color="grey",alpha=0.8)
        style_axes(axs[i,j])


# axs[1,0].set_xlabel("Iteration number $i$")
axs[1,1].set_xlabel("Iteration number $i$",fontsize=12)
axs[0,0].set_ylabel("Measurement space\n misfit",fontsize=12)
axs[1,0].set_ylabel("State space\n error",fontsize=12)
# axs[0,0].set_title("Deterministic EKI")
axs[0,1].set_title("(Large ensemble)")
axs[0,2].set_title("(Small ensemble)")
fig.text(0.6, 0.95, 'Stochastic EKI', ha='center', fontsize=13)
fig.text(0.23, 0.95, 'Deterministic EKI', ha='center', fontsize=13)


# legend for measurement space row
axs[0,2].legend(lines[0],lbls[0],loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,fontsize=12)

# legend for state space row
axs[1, 2].legend(lines[1],lbls[1],loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,fontsize=12)

# Legend for a 1/sqrt(i) rate line
proxy_line = Line2D([0], [0], color='gray', alpha=0.8, label="$1/\\sqrt{i}$ rate")
fig.legend(handles=[proxy_line], loc='center left', bbox_to_anchor=(0.8, 0.5), frameon=False,fontsize=12)

# plt.tight_layout()
plt.subplots_adjust(top=0.85,right=0.8,left=0.13,bottom=0.15)
fig.savefig("temp.pdf")
plt.close()
