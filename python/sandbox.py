import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from linearEKI import *

###########################################################################
# CODE TO SET UP AND RUN EKI
###########################################################################
# SETUP LEAST SQUARE PROBLEM
# construct a random 3x3 H with rank 2
basis,_ = np.linalg.qr(np.random.rand(3,3))
H = np.hstack((basis[:,:2],np.sum(basis[:,:2],axis=1)[:,np.newaxis])).T
meas = np.array([0.1, 0.2, 0.3])[:,np.newaxis]
prob = leastsquares(H=H)

# SETUP EKI ITERATION
# construct a random rank 2 ensemble with one component in range of H and one not in ran(H)
J = 15
th = 2*np.pi*np.random.rand(1,J)
v0 = basis[:,[0, 2]] @ np.vstack((np.cos(th),np.sin(th)))

maxiter = 100
det = EKI(prob,"det",maxiter,v0 = v0)
stoch = EKI(prob,"stoch",maxiter,v0 = det.v0)


###########################################################################
# Plotting
###########################################################################

# define colors
orange = "#FFB320"
blue   = "#5BA9EF"
cobalt = "#214ac4"
maroon = "#a62216"
persimmon = "#e8682c"
lime = "#86d631"


fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=("State Space", "Measurement Space"),
    horizontal_spacing = 0.1
)

# Define a function to create line traces with end annotations
def create_line_trace(x, y, z, name, color):
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+text',
        name=name,
        line=dict(color=color),
        text=[None, name],
        textposition='top center',
        textfont=dict(color=color)
    )

def plot_vector3(q, name, color):
    return go.Scatter3d(
        x=[0, q[0]], y=[0,q[1]], z=[0,q[2]],
        mode='lines+text',
        name=name,
        line=dict(color=color),
        text=[None, name],
        textposition='top center',
        showlegend=False,
        textfont=dict(color=color)
    )

def plot_point3(q, name, color,sym="circle"):
    return go.Scatter3d(
        x=[q[0]], y=[q[1]], z=[q[2]],
        mode='markers+text',
        # name=name,
        marker=dict(size=8, color="black", opacity=0.8,symbol=sym),
        text=[name],
        textposition='top center',
        showlegend=False,
        textfont=dict(color=color)
    )

def plot_plane(u, v, annotation_text,grid_size=1, color = "gray", label = None,num_points=100):
    # Define the grid for the plane
    x = np.linspace(-grid_size, grid_size, num_points)
    y = np.linspace(-grid_size, grid_size, num_points)
    x, y = np.meshgrid(x, y)

    # Define the plane using the basis vectors
    z = u[2] * x + v[2] * y

    # Transform the grid using the basis vectors
    transformed_x = u[0] * x + v[0] * y
    transformed_y = u[1] * x + v[1] * y
    transformed_z = z

    # Add the 2D plane
    scl = [[0, color], [1, color]]
    plane = go.Surface(z=transformed_z, x=transformed_x, y=transformed_y, colorscale=scl, opacity=0.5,name=label,showscale=False,showlegend=False)

    text = go.Scatter3d(
        # x=[(transformed_x[0,0] + transformed_x[-1,-1]) / 2],
        # y=[(transformed_y[0,0] + transformed_y[-1,-1]) / 2],
        # z = [(transformed_z[0,0] + transformed_z[-1,-1]) / 2],
        x=[transformed_x[0,0]],
        y=[transformed_y[0,0]],
        z = [transformed_z[0,0]],
        text=[annotation_text],
        mode='text',
        # marker=dict(size=8, color="red", opacity=0.8),
        textfont=dict(color=color),
        showlegend=False
    )
    return plane, text

# Data for state space

# get unit basis vectors in range of all 3 projectors and plot them 
temp = np.random.rand(3,1)
q1 = det.bbPr @ temp
q1 = np.squeeze(q1/np.linalg.norm(q1))
q2 = det.bbQr @ temp
q2 = np.squeeze(q2/np.linalg.norm(q2))
q3 = det.bbNr @ temp
q3 = np.squeeze(q3/np.linalg.norm(q3))

lines1 = [
    plot_vector3(q1,"Ran(Pr)",cobalt),
    plot_vector3(q2,"Ran(Qr)",persimmon),
    plot_vector3(q3,"Ran(Nr)",maroon)
]
for trace in lines1:
    fig.add_trace(trace, row=1, col=1)
x = np.squeeze(det.vv[:,0,:])
y = np.squeeze(det.vv[:,1,:])
z = np.squeeze(det.vv[:,2,:])
vs = np.squeeze(det.ls.vstar)
for j in range(J):
    if j == 0:
        sl = True
    else:
        sl = False
    fig.add_trace(go.Scatter3d(x=x[:,j], y=y[:,j], z=z[:,j], mode='lines',line=dict(width=4, color=blue),name="Particle paths",showlegend=sl),row=1,col=1)
    fig.add_trace(go.Scatter3d(x=[x[-1,j]], y=[y[-1,j]], z=[z[-1,j]], mode='markers',marker=dict(size=5, color=blue, opacity=0.8),name="path end",showlegend=sl),row=1,col=1)
pln,txt = plot_plane(basis[:,0], basis[:,2], "Ran(Gamma_i)",grid_size=1, color = blue)
fig.add_trace(pln,row=1,col=1)
fig.add_trace(txt,row=1,col=1)
fig.add_trace(plot_point3(vs, "v*", "black",sym="cross"),row=1,col=1)
pln,txt = plot_plane(basis[:,0], basis[:,1], "Ran(H^T)",grid_size=1, color = orange)
fig.add_trace(pln,row=1,col=1)
fig.add_trace(txt,row=1,col=1)

# Data for the second scatter plot
# get unit basis vectors in range of all 3 projectors and plot them 
temp = np.random.rand(3,1)
q1 = det.calPr @ temp
q1 = np.squeeze(q1/np.linalg.norm(q1))
q2 = det.calQr @ temp
q2 = np.squeeze(q2/np.linalg.norm(q2))
q3 = det.calNr @ temp
q3 = np.squeeze(q3/np.linalg.norm(q3))

lines1 = [
    plot_vector3(q1,"Ran(Pr)",cobalt),
    plot_vector3(q2,"Ran(Qr)",persimmon),
    plot_vector3(q3,"Ran(Nr)",maroon)
]
for trace in lines1:
    fig.add_trace(trace, row=1, col=2)
hh = det.ls.H[np.newaxis,:,:] @ det.vv

x = np.squeeze(hh[:,0,:])
y = np.squeeze(hh[:,1,:])
z = np.squeeze(hh[:,2,:])
meas = np.squeeze(det.ls.meas)
for j in range(J):
    fig.add_trace(go.Scatter3d(x=x[:,j], y=y[:,j], z=z[:,j], mode='lines',line=dict(width=4, color=blue),name="Particle paths",showlegend=False),row=1,col=2)
    fig.add_trace(go.Scatter3d(x=[x[-1,j]], y=[y[-1,j]], z=[z[-1,j]], mode='markers',marker=dict(size=5, color=blue, opacity=0.8),name="path end",showlegend=False),row=1,col=2)
fig.add_trace(plot_point3(meas, "m", "black",sym="cross"),row=1,col=2)

# add range of H
b1 = det.ls.H[:,0]
b1 = np.squeeze(b1/np.linalg.norm(b1))
b2 = np.squeeze(det.ls.H[:,1])
b2 = b2 - ((b1.T @ b2) * b1)
b2 = np.squeeze(b2/np.linalg.norm(b2))
pln,txt = plot_plane(b1,b2, "Ran(H)",grid_size=1, color = orange)
fig.add_trace(pln,row=1,col=2)
fig.add_trace(txt,row=1,col=2)


fig.update_layout(
        height=600,  # Adjust height as needed
        showlegend=True,
        legend=dict(
            x=0.5,
            y=0.5,
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
        )
    )

pio.write_html(fig, file='subspaces3D.html', auto_open=True)
fig.show()
