import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from linearEKI import *
import pickle

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

with open("seed2_5000.pkl","rb") as f:
    eki_runs = pickle.load(f)

detlarge   = eki_runs[0]
detsmall   = eki_runs[1]
stochsmall = eki_runs[2]
stochlarge = eki_runs[3]

n,d     = detlarge.ls.H.shape
Jsmall  = detsmall.v.shape[1]
Jlarge  = detlarge.v.shape[1]
maxiter = detlarge.components.shape[0]-1

orange = "#E97132"
blue = "#6ABCEB"
black  = "#888888"
colors = [blue,orange,black]
styles = ["solid","dashed","dotted"]

cols = [detlarge, stochlarge,detsmall,stochsmall]
coltitles = ["Deterministic","Stochastic","Deterministic", "Stochastic"]
rows = ["misfit","error"]
projs = [["calP","calQ","calN"], ["bbP","bbQ","bbN"]]
lbls  = [["$\\|\\boldsymbol{\\mathcal{P}}\\boldsymbol{\\theta}_i^{(j)}\\|$","$\\|\\boldsymbol{\\mathcal{Q}}\\boldsymbol{\\theta}_i^{(j)}\\|$","$\\|\\boldsymbol{\\mathcal{N}}\\boldsymbol{\\theta}_i^{(j)}\\|$"],["$\\|\\mathbf{P}\\boldsymbol{\\omega}_i^{(j)}\\|$","$\\|\\mathbf{Q}\\boldsymbol{\\omega}_i^{(j)}\\|$","$\\|\\mathbf{N}\\boldsymbol{\\omega}_i^{(j)}\\|$"]]

lines = [[],[]]

fig, axs = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(5.5,4))

x = np.arange(maxiter+1)
xx = np.linspace(1.0001, maxiter+1)

for i in range(2): # row
    for j in range(4): # column
        for k in [2,1,0]: # linetype
            y = cols[j].components[:,i*3+k,:50]
            if i == 0:
                scl = 50
            else:
                scl = 5
            ln = axs[i,j].loglog(x,y,alpha=0.3,color=colors[k],linestyle=styles[k],label=lbls[i][k])

            if j == 0:
                lines[i].append(ln[0])
        sqrt = axs[i,j].loglog(xx,scl/np.sqrt(xx),color="#555555",alpha=0.8)
        style_axes(axs[i,j])


fig.text(0.55,0.01,"Iteration number $i$",fontsize=12,ha='center')
axs[0,0].set_ylabel("Measurement misfit",fontsize=12,labelpad=8)
axs[1,0].set_ylabel("State residual",fontsize=12,labelpad=1)
for i in range(4):
    axs[0,i].set_title(coltitles[i],pad=1,fontsize=10.5)
fig.text(0.76, 0.95, 'Small ensemble', ha='center', fontsize=13)
fig.text(0.34, 0.95, 'Large ensemble', ha='center', fontsize=13)

# # legend for measurement space row
lines_all = [lines[0][2],lines[1][2],lines[0][1],lines[1][1],lines[0][0],lines[1][0]]
lbls_all  = [lbls[0][0],lbls[1][0],lbls[0][1],lbls[1][1],lbls[0][2],lbls[1][2]]
axs[0,0].legend(lines_all,lbls_all,loc='lower left',bbox_to_anchor=(-0.1,-0.7),ncols=3,frameon=False,fontsize=12,handletextpad=0.2,columnspacing = 1)

# Legend for a 1/sqrt(i) rate line
proxy_line = Line2D([0], [0], color='#555555', alpha=0.8, label="$1/\\sqrt{i}$ rate")
fig.legend(handles=[proxy_line], loc='center left', bbox_to_anchor=(0.75, 0.485), frameon=False,handletextpad=0.2,fontsize=12)

plt.subplots_adjust(top=0.88,right=0.98,left=0.13,bottom=0.1,hspace=0.7,wspace=0.1)
fig.savefig("paths.pdf")
plt.close()
