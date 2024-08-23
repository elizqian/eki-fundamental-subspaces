import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np

# define colors
darkblue   = "#1895de"
cobalt = "#214ac4"
maroon = "#a62216"
persimmon = "#e8682c"
lime = "#86d631"
orange = "#E97132"
blue = "#6ABCEB"
black  = "#222222"

# function to plot vectors with annotations
def plot_vector3(fig,r,c,q, name, color):
    fig.add_trace(go.Scatter3d(
        x=[-q[0], q[0]], y=[-q[1],q[1]], z=[-q[2],q[2]],
        mode='lines+text',
        name=name,
        line=dict(color=color,width = 3),
        text=[None, name],
        textposition='top center',
        showlegend=False,
        textfont=dict(color=color)
    ),row=r,col=c)

# function to plot points with labels
def plot_point3(fig,r,c,q, name, color,sym="circle"):
    fig.add_trace(go.Scatter3d(
        x=[q[0]], y=[q[1]], z=[q[2]],
        mode='markers+text',
        marker=dict(size=8, color=color, opacity=0.8,symbol=sym),
        text=[name],
        textposition='top center',
        showlegend=False,
        textfont=dict(color=color)
    ),row=r,col=c)

def plot_star3(fig,r,c,center,name,color,b1,b2):
    # orthogonalize basis
    b1 = b1/np.linalg.norm(b1)
    b2 = b2 - (b1.T @ b2)* b1 
    b2 = b2/np.linalg.norm(b2)

    # define coordinates of star
    scl = 0.1
    outer_radius = scl*1
    inner_radius = scl*0.4
    n_points = 5
    angles = np.linspace(0, 2 * np.pi, 2 * n_points + 1)
    points = []
    for i, angle in enumerate(angles):
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + radius * (b1[0] * np.cos(angle) + b2[0] * np.sin(angle))
        y = center[1] + radius * (b1[1] * np.cos(angle) + b2[1] * np.sin(angle))
        z = center[2] + radius * (b1[2] * np.cos(angle) + b2[2] * np.sin(angle))
        points.append([x, y, z])
    x_coords, y_coords, z_coords = zip(*points)
    
    # add star to plot
    fig.add_trace(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='lines', line=dict(width=3,color=color),showlegend=False),row=r,col=c)
    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode='text',
        text=[name],
        textposition='top center',
        showlegend=False,
        textfont=dict(color=color)
    ),row=r,col=c)

# function to plot planes from basis vectors with annotations
def plot_plane(fig,r,c,u, v, annotation_text,grid_size=1, color = "gray", label = None,num_points=100):
    x = np.linspace(-grid_size, grid_size, num_points)
    y = np.linspace(-grid_size, grid_size, num_points)
    x, y = np.meshgrid(x, y)

    z = u[2] * x + v[2] * y
    transformed_x = u[0] * x + v[0] * y
    transformed_y = u[1] * x + v[1] * y
    transformed_z = z

    scl = [[0, color], [1, color]]
    fig.add_trace(go.Surface(z=transformed_z, x=transformed_x, y=transformed_y, colorscale=scl, opacity=0.5,name=label,showscale=False,showlegend=False),row=r,col=c)

    fig.add_trace(go.Scatter3d(
        x=[transformed_x[0,0]],
        y=[transformed_y[0,0]],
        z = [transformed_z[0,0]],
        text=[annotation_text],
        mode='text',
        textfont=dict(color=color),
        showlegend=False
    ),row=r,col=c)

def plot_paths(fig,r,c,paths,color,name):
    _,_,J = paths.shape
    x = np.squeeze(paths[:,0,:])
    y = np.squeeze(paths[:,1,:])
    z = np.squeeze(paths[:,2,:])
    for j in range(J):
        if j == 1:
            sl = True
        else:
            sl = False
        line = go.Scatter3d(x=x[:,j], y=y[:,j], z=z[:,j], mode='lines',line=dict(width=3, color=color,dash="dot"),name=name,showlegend=False)
        u = x[-1,j] - x[-2,j]
        v = y[-1,j] - y[-2,j]
        w = z[-1,j] - z[-2,j]
        arrow = go.Cone(
            x=[x[-1,j]], y=[y[-1,j]], z=[z[-1,j]],
            u=[u], v=[v], w=[w],
            sizemode="absolute",
            sizeref=0.1,
            showscale=False,
            colorscale=[[0, color], [1, color]]
        )
        fig.add_trace(line,row=r,col=c)
        fig.add_trace(arrow,row=r,col=c)
