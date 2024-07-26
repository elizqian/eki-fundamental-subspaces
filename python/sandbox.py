import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.cm as cm
from linearEKI import *

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('darkgrey')
    ax.spines['left'].set_color('darkgrey')
    ax.tick_params(axis='both', which='both', colors='darkgrey')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')

def plot_2d_plane(fig,idx,u, v, grid_size=1, col = "gray", label = None,num_points=100):
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
    scl = [[0, col], [1, col]]
    fig.add_trace(go.Surface(z=transformed_z, x=transformed_x, y=transformed_y, colorscale=scl, opacity=0.5,name=label,showscale=False,showlegend=True),row=1,col=idx)

def plot_line(existing_fig,idx, basis_vector, line_name='Line', color='red', num_points=100):
    """
    Plots a line in 3D space given a basis vector.

    Args:
    basis_vector (list or np.array): Basis vector in the form [x, y, z].
    existing_fig (go.Figure): Existing Plotly figure to add the line to.
    line_name (str): Name of the line for the legend.
    color (str): Color of the line.
    num_points (int): Number of points along the line.
    """
    # Generate points along the line
    t = np.linspace(-1, 1, num_points)
    x = basis_vector[0] * t
    y = basis_vector[1] * t
    z = basis_vector[2] * t

    # Add the line to the existing figure
    existing_fig.add_trace(go.Scatter3d(
        x=x, 
        y=y, 
        z=z, 
        mode='lines',
        line=dict(color=color, width=3),
        name=line_name,
        showlegend=True
    ),row=1,col=idx)

# SETUP LEAST SQUSRE PROBLEM
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

# PREP DATA FOR PLOTTING
x = np.squeeze(det.vv[:,0,:])
y = np.squeeze(det.vv[:,1,:])
z = np.squeeze(det.vv[:,2,:])

temp = np.random.rand(3,1)
q1 = det.bbPr @ temp
q1 = q1/np.linalg.norm(q1)
q2 = det.bbQr @ temp
q2 = q2/np.linalg.norm(q2)
q3 = det.bbNr @ temp
q3 = q3/np.linalg.norm(q3)

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    subplot_titles=("State Space", "Measurement Space")
)

orange = "#FFB320"
blue   = "#5BA9EF"

vs = det.ls.vstar
for j in range(J):
    # plot ensemble
    if j == 0:
        sl = True
    else:
        sl = False
    fig.add_trace(go.Scatter3d(x=x[:,j], y=y[:,j], z=z[:,j], mode='lines',line=dict(width=4, color=blue),name="Particle paths",showlegend=sl))
    fig.add_trace(go.Scatter3d(x=[x[-1,j]], y=[y[-1,j]], z=[z[-1,j]], mode='markers',marker=dict(size=5, color=blue, opacity=0.8),name="path end",showlegend=sl))
fig.add_trace(go.Scatter3d(x=vs[0], y=vs[1], z=vs[2], mode='markers',marker=dict(size=5, color="black", opacity=0.8),name=r"$v^*$"))

plot_2d_plane(fig,1,basis[:,0],basis[:,2],grid_size = 1.5,col=blue,label = r"$\textsf{Ran}(\Gamma_i)$")
plot_2d_plane(fig,1,basis[:,0],basis[:,1],grid_size = 1.5,label = r"$\textsf{Ran}(H^\top)$")
plot_line(fig,1,q1,line_name=r"$\textsf{Ran}(\mathbb{P}_r)$",color="#214ac4")
plot_line(fig,1,q2,line_name=r"$\textsf{Ran}(\mathbb{Q}_r)$",color="#e89220")
plot_line(fig,1,q3,line_name=r"$\textsf{Ran}(\mathbb{N}_r)$",color="#a62216")
# Update layout for better visualization
fig.update_layout(scene=dict(
    xaxis_title='X Axis',
    yaxis_title='Y Axis',
    zaxis_title='Z Axis'
))
fig.show()