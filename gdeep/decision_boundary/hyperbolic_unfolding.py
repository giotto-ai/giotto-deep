import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd


class Geodesics():
    """
    Standard Data type of geodesics
    """
    def __init__(self, geodesics_tensor):
        self.geodesics_tensor = geodesics_tensor
        self.dim = int(geodesics_tensor.shape[-1]/2)
        self.n_geodesics = geodesics_tensor.shape[1]
        
    def get_trajectories(self):
        return self.geodesics_tensor[:,:,:self.dim]
    
    def get_endpoints(self):
        return self.geodesics_tensor[-1,:,:self.dim]
    
    def get_dim(self):
        return self.dim
    
    def plot_trajectories(self, plot_file=None, show=True):
        try:
            assert(self.dim==2 or self.dim==3)
        except:
            raise ValueError(f'Only works in 2 and 3 dimension up to now!')
        elif self.dim==3:
            length_trajectories = self.trajectories().shape[0]
            df_trajectories = pd.DataFrame(np.einsum('ijk->jik',self.trajectories()).reshape(-1,3), columns=["x"+str(i) for i in range(3)])
            df_trajectories['index'] = [int(i/length_trajectories) for i in range(df_trajectories.shape[0])]
            
            fig = px.line_3d(df_trajectories, x="x0", y="x1", z="x2", color='index')
            if show:
                fig.show()
            else:
                return fig
            

                
    def plot_endpoints_with_dataset(self, df_dataset, plot_file: bool=None, filter_fkt=None, verbose=False, show=True):
        try:
            assert(self.dim==2 or self.dim==3)
        except:
            raise ValueError('Only works in dimension 2 and 3 up to now!')

        if self.dim==2:
            endpoint = self.endpoints()
            if filter_fkt:
                endpoints_filtered = filter_fkt(endpoint)
            else:
                endpoints_filtered = endpoint
            if verbose:
                print(f"Shape of endpoints_filtered: {endpoints_filtered.shape}")
            df_endpoints = pd.DataFrame(endpoints_filtered, columns=["x"+str(i) for i in range(2)])
            df_endpoints["label"] = [0.5 for i in range(endpoints_filtered.shape[0])]
            df_dataset = df_dataset.rename(columns={"x": "x0", "y": "x1"})
            fig = px.scatter(pd.concat([df_endpoints, df_dataset]), x="x0", y="x1", color="label")


        elif self.dim==3:
            endpoint = self.endpoints()
            if filter_fkt:
                endpoints_filtered = filter_fkt(endpoint)
            else:
                endpoints_filtered = endpoint
            if verbose:
                print(f"Shape of endpoints_filtered: {endpoints_filtered.shape}")
            df_endpoints = pd.DataFrame(endpoints_filtered, columns=["x"+str(i) for i in range(3)])
            df_endpoints["label"] = [3 for i in range(endpoints_filtered.shape[0])]
            df_dataset = df_dataset.rename(columns={"x": "x0", "y": "x1", "z": "x2"})
            fig = px.scatter_3d(pd.concat([df_endpoints, df_dataset]), x="x0", y="x1", z="x2", color="label")
            fig.show()
            if plot_file:
                try:
                    fig.write_html(os.path.join('plots', plot_file))
                except:
                    os.mkdir('plots')
                    fig.write_html(os.path.join('plots', plot_file))



class FlatEuclidean(nn.Module):
    def __init__(self, dim: int = 2):
        super().__init__()
        
        self.dim = dim

                
    def forward(self, t, y):
        try:
            assert(y.shape[-1]==2*self.dim)
        except:
            raise ValueError(f'input has to be a {2*self.dim}-dimensional vector')
        
        dy = y[:,-self.dim:]
        ddy = torch.zeros_like(y[:,-self.dim:])
        
        
        return torch.cat((dy,ddy),-1)
    
    
class TwoSphere(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dim = 2

                
    def forward(self, t, y):
        try:
            assert(y.shape[-1]==2*self.dim)
        except:
            raise ValueError(f'input has to be a {2*self.dim}-dimensional vector')
        
        # y = [\theta, \phi, d\theta, d\phi]
        # dy = [d\theta, d\phi]
        
        dy = y[:,-self.dim:]
        ddy = torch.zeros_like(y[:,-self.dim:])

        ddy[:,0] = torch.sin(y[:,0]) * torch.cos(y[:,0]) * y[:,3]**2
        ddy[:,1] = -2*torch.cos(y[:,0])/torch.sin(y[:,0]) * y[:,2] * y[:,3]

        
        return torch.cat((dy,ddy),-1)
    
class UpperHalfPlane(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dim = 2

                
    def forward(self, t, y):
        try:
            assert(y.shape[-1]==2*self.dim)
        except:
            raise ValueError(f'input has to be a {2*self.dim}-dimensional vector')
        
        dy = y[:,-self.dim:]
        ddy = torch.zeros_like(y[:,-self.dim:])
        
        ddy[:,0] = 2/y[:,1]*y[:,2]*y[:,3]
        ddy[:,1] = -1/y[:,1]*(y[:,2]**2-y[:,3]**2)
        
        
        return torch.cat((dy,ddy),-1)




class CircleNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dim = 2

                
    def forward(self, x_cat, x_cont):
        try:
            assert(x_cont.shape[-1]==2)
        except:
            raise ValueError(f'input has to be a {2}-dimensional vector')
        activation = 0.5*torch.exp(-torch.sum(x_cont**2, axis=-1)+1)
        return activation.reshape((-1,1))
    
    def return_input_dim(self):
        return 2

class ConformTrafoNN(nn.Module):
    def __init__(self, nn: nn.Module, input_dim: int):
        super().__init__()
        
        self.nn = nn
        self.input_dim = input_dim

    def forward(self, x_cat, x_cont):
        return self.nn.forward(x_cat, x_cont)
    
    def conform_factor(self, x):
        return torch.sum(torch.abs(self.forward(None, x)-0.5), axis=-1)
    
    def gradient(self, x):
        delta = torch.zeros_like(x, requires_grad=True)
        conform_trans = torch.log(torch.abs(self.forward(None, x+delta)-0.5))
        torch.sum(conform_trans).backward()
        return delta.grad.detach()

    def return_input_dim(self):
        return self.input_dim
    

class HyperbolicUnfoldingGeoEq(nn.Module):
    def __init__(self, nn: ConformTrafoNN):
        super().__init__()
        
        self.nn = nn
        self.input_dim = self.nn.return_input_dim()
        self.dim = self.input_dim

                
    def forward(self, t, y):
        try:
            assert(y.shape[-1]==2*self.input_dim)
        except:
            raise ValueError(f'input has to be a {2*self.input_dim}-dimensional vector not {y.shape[-1]}')
        
        # y = [y,dy]
        
        dy = y[:,-self.input_dim:]
        y = y[:, :self.input_dim]
        
        gradient_log_delta = self.nn.gradient(y)
        
        # quasi-hyperbolic geodesic equation see markdown comment
        ddy = 2*torch.einsum('bi,bi,bj->bj', gradient_log_delta, dy, dy)\
              - torch.einsum('bi,bj,bj->bi', gradient_log_delta, dy, dy)
        return torch.cat((dy,ddy),-1)