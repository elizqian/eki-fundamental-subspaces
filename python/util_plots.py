import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np

# define colors
orange = "#FFB320"
blue   = "#5BA9EF"
cobalt = "#214ac4"
maroon = "#a62216"
persimmon = "#e8682c"
lime = "#86d631"

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
        marker=dict(size=8, color="black", opacity=0.8,symbol=sym),
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
        if j == 0:
            sl = True
        else:
            sl = False
        line = go.Scatter3d(x=x[:,j], y=y[:,j], z=z[:,j], mode='lines',line=dict(width=2, color=color),name=name,showlegend=sl)
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
