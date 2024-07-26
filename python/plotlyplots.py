import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from linearEKI import *

pio.renderers.default = "vscode"



prob = leastsquares()

maxiter = 1000
det = EKI(prob,"det",maxiter)
stoch = EKI(prob,"stoch",maxiter,v0 = det.v0)

measPD = det.getComponentNorm("misfit","calPr")
measQD = det.getComponentNorm("misfit","calQr")
measND = det.getComponentNorm("misfit","calNr")

statePD = det.getComponentNorm("error","bbPr")
stateQD = det.getComponentNorm("error","bbQr")
stateND = det.getComponentNorm("error","bbNr")

measPS = stoch.getComponentNorm("misfit","calPr")
measQS = stoch.getComponentNorm("misfit","calQr")
measNS = stoch.getComponentNorm("misfit","calNr")
statePS = stoch.getComponentNorm("error","bbPr")
stateQS = stoch.getComponentNorm("error","bbQr")
stateNS = stoch.getComponentNorm("error","bbNr")

scl = np.zeros((2,2))
scl[0,0] = np.max(statePD[1,:])
scl[0,1] = np.max(statePS[1,:])
scl[1,0] = np.max(measPD[1,:])
scl[1,1] = np.max(measPS[1,:])

# Generate sample data
x = np.arange(maxiter+1)
y1 = x ** 2
y2 = x ** 1.5
y3 = x ** 1
y4 = x ** 0.5

# Create 2x2 subplot
fig = make_subplots(rows=2, cols=2, subplot_titles=('Deterministic EKI', 'Stochastic EKI', '', ''),shared_yaxes = True,shared_xaxes=True,vertical_spacing=0.05,horizontal_spacing=0.05)

blue = 'rgba(30, 126, 229, 0.2)'
orange = 'rgba(251,167,30, 0.2)'
colors = ['rgba(30, 126, 229, 0.2)', 'rgba(251,167,30, 0.2)', 'rgba(0, 0, 255, 0.6)']

for j in range(det.J):
    if j == 0:
        sl = True 
    else:
        sl = False
    fig.add_trace(go.Scatter(x=x, y=statePD[:,j], mode='lines',line=dict(color=blue), name=r'$\large \left\|\mathbb{P}_r\omega_i^{(j)}\right\|$',showlegend=sl), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=stateQD[:,j], mode='lines',line=dict(color=blue,dash = "dash"), name=r'$\large \left\|\mathbb{Q}_r\omega_i^{(j)}\right\|$',showlegend=sl), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=stateND[:,j], mode='lines',line=dict(color=blue,dash = "dot"), name=r'$\large \left\|\mathbb{N}_r\omega_i^{(j)}\right\|$',showlegend=sl), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=statePS[:,j], mode='lines', line=dict(color=blue),showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=stateQS[:,j], mode='lines', line=dict(color=blue,dash = "dash"),showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=stateQS[:,j], mode='lines', line=dict(color=blue,dash = "dot"),showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(x=x, y=measPD[:,j], mode='lines',line=dict(color=orange), name=r'$\large \left\|\mathcal{P}_r\theta_i^{(j)}\right\|$',showlegend=sl), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=measQD[:,j], mode='lines',line=dict(color=orange,dash = "dash"), name=r'$\large \left\|\mathcal{Q}_r\theta_i^{(j)}\right\|$',showlegend=sl), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=measND[:,j], mode='lines',line=dict(color=orange,dash = "dot"),name=r'$\large \left\|\mathcal{N}_r\theta_i^{(j)}\right\|$',showlegend=sl), row=2, col=1)

    fig.add_trace(go.Scatter(x=x, y=measPS[:,j], mode='lines',line=dict(color=orange),showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x, y=measQS[:,j], mode='lines',line=dict(color=orange,dash = "dash"),showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x, y=measNS[:,j], mode='lines',line=dict(color=orange,dash = "dot"),showlegend=False), row=2, col=2)

# Set log-log axes
for i in range(1, 3):
    for j in range(1, 3):
        if i == 1 and j == 1:
            sl = True
        else:
            sl = False
        fig.add_trace(go.Scatter(x=x,y=scl[i-1,j-1]/np.sqrt(x),mode='lines',line=dict(color="grey",dash="solid"),name=r'$\large 1/\sqrt{i} \text{ rate}$',showlegend=sl),row=i,col=j)
        fig.update_xaxes(type='log', row=i, col=j, showgrid=True, gridcolor='lightgrey',showline=True, linecolor='lightgrey',exponentformat="power",dtick=1)
        fig.update_yaxes(type='log', row=i, col=j, showgrid=True, gridcolor='lightgrey',showline=True, linecolor='lightgrey',exponentformat="power",dtick=1)


fig.update_xaxes(title_text=r'$\large \text{Iteration  index}\,\,\, i$', row=2, col=1)
fig.update_xaxes(title_text=r'$\large \text{Iteration  index}\,\,\, i$', row=2, col=2)

# Update layout to set background to white
fig.update_layout(
    # title_text="2x2 Log-Log Subplots",
    showlegend=True,
    plot_bgcolor='white',
    paper_bgcolor='white', 
    height = 1.2*4*96,
    width = 1.2*5.5*96,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(
        # orientation = "h",
        # x=0.5,  # Position legend outside the plot
        y=0.5,
        # xanchor="center",
        yanchor="middle",
        font=dict(
            family = 'Computer Modern',
            weight='normal',
            size = 48,
        ),
    )
)

# # Update layoutpi
# fig.update_layout(title_text="2x2 Log-Log Subplots", showlegend=False)

# Show plot
fig.write_image("convrates_plotly.pdf")
