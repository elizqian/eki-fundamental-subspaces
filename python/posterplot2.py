import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
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
det = EKI(prob,"det",maxiter,v0 = v0)          # deterministic
stochsmall = EKI(prob,"stoch",maxiter,v0 = v0small) # stochastic small ensemble
stochlarge = EKI(prob,"stoch",maxiter,v0=v0)        # stoch large ensemble


##############################
# plotting setup and plot
##############################

font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
lato_font_path = [path for path in font_paths if 'Lato' in path]
for lfp in lato_font_path:  
    fm.fontManager.addfont(lfp)

plt.rcParams['font.family'] = 'Lato'
plt.rcParams['text.usetex'] = True         # Use LaTeX for math, 
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts,amsmath}'
# plt.rcParams['mathtext.fontset'] = 'cm'


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('darkgrey')
    ax.spines['left'].set_color('darkgrey')
    ax.tick_params(axis='both', which='both', colors='darkgrey',labelsize=54)
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')

orange = "#E97132"
blue = "#6ABCEB"
black  = "#888888"
colors = [blue,orange,black]
styles = ["solid","dashed","dotted"]

rows = [det, stochlarge,stochsmall]
cols = ["misfit","error"]
projs = [["calP","calQ","calN"], ["bbP","bbQ","bbN"]]
lbls  = [["$\\|\\boldsymbol{\\mathcal{P}}\\boldsymbol{\\theta}_i^{(j)}\\|$","$\\|\\boldsymbol{\\mathcal{Q}}\\boldsymbol{\\theta}_i^{(j)}\\|$","$\\|\\boldsymbol{\\mathcal{N}}\\boldsymbol{\\theta}_i^{(j)}\\|$"],["$\\|\\mathbb{P}\\boldsymbol{\\omega}_i^{(j)}\\|$","$\\|\\mathbb{Q}\\boldsymbol{\\omega}_i^{(j)}\\|$","$\\|\\mathbb{N}\\boldsymbol{\\omega}_i^{(j)}\\|$"]]

lines = [[],[]]

#############################################
# 2x2 simplified
#############################################
# font_path = '/usr/share/fonts/truetype/lato/Lato-Regular.ttf' 
# latoLabel = fm.FontProperties(fname=font_path,size=72)
fig, axs = plt.subplots(2,2, sharex='col', sharey='row', figsize=(24,18))

x = np.arange(maxiter+1)
xx = np.linspace(1.0001, maxiter+1)

for i in range(2): # col
    for j in range(2): # row
        for k in [1,2,0]: # linetype
            y = rows[j].getComponentNorm(cols[i],projs[i][k])
            if k == 0:
                scl = np.max(y[1,:])
            ln = axs[j,i].loglog(x,y,alpha=0.3,color=colors[k],linestyle=styles[k],label=lbls[i][k],linewidth=8)
            prox = Line2D([0], [0], color=colors[k],linestyle=styles[k], label=lbls[i][k],linewidth=8)
            if j == 0:
                lines[i].append(prox)
        sqrt = axs[j,i].loglog(xx,scl/np.sqrt(xx),color="#555555",alpha=0.8,linewidth=8)
        style_axes(axs[j,i])

axs[1,0].set_xlabel("Iteration number $i$",fontsize=60,labelpad=2)
axs[1,1].set_xlabel("Iteration number $i$",fontsize=60,labelpad=2)
axs[0,0].set_title("Data misfit",pad=130,fontsize=72,x=0.4)
axs[0,1].set_title("Least squares residual",fontsize=72,pad=130)

axs[0,0].set_ylabel("Deterministic EKI",fontsize=60)
axs[1,0].set_ylabel("Stochastic EKI",fontsize=60)

# legend
lines_all = [lines[0][2],lines[0][0],lines[0][1],lines[1][2],lines[1][0],lines[1][1]]
lbls_all  = [lbls[0][0],lbls[0][1],lbls[0][2],lbls[1][0],lbls[1][1],lbls[1][2]]

axs[0,0].legend([lines[0][2],lines[0][0],lines[0][1]],[lbls[0][0],lbls[0][1],lbls[0][2]],frameon=False,loc='upper left',fontsize=54,handletextpad=0.1,bbox_to_anchor=(-0.3,1.3),ncols=3,columnspacing=0.1,handlelength=1.25)

axs[0,1].legend([lines[1][2],lines[1][0],lines[1][1]],[lbls[1][0],lbls[1][1],lbls[1][2]],frameon=False,loc='upper left',fontsize=54,handletextpad=0.1,bbox_to_anchor=(-0.15,1.3),ncols=3,columnspacing=0.1,handlelength=1.25)

# # Legend for a 1/sqrt(i) rate line
proxy_line = Line2D([0], [0], color='#555555', alpha=0.8,linewidth=8, label="$1/\\sqrt{i}$ rate")
fig.legend(handles=[proxy_line], loc='center left', bbox_to_anchor=(0.55, 0.58), frameon=False,handletextpad=0.2,fontsize=54)
fig.legend(handles=[proxy_line], loc='center left', bbox_to_anchor=(0.12, 0.58), frameon=False,handletextpad=0.2,fontsize=54)

plt.subplots_adjust(top=0.85,right=0.97,left=0.12,bottom=0.08,hspace=0.1,wspace=0.1)
# plt.tight_layout()
fig.savefig("posterResults22.pdf")
plt.close()