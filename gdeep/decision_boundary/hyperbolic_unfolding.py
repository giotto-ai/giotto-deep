import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



#TODO: refactor into smaller classes
class Geodesics():
    """
    Standard Data type of geodesics
    """
    def __init__(self, geodesics_tensor):
        self.geodesics_tensor = geodesics_tensor
        self.dim = int(geodesics_tensor.shape[-1]/2)
        self.n_geodesics = geodesics_tensor.shape[1]
        
    def trajectories(self):
        return self.geodesics_tensor[:,:,:self.dim]
    
    def endpoints(self):
        return self.geodesics_tensor[-1,:,:self.dim]
    
    
    def plot_trajectories(self, plot_file=None):
        try:
            assert(self.dim==2 or self.dim==3)
        except:
            raise ValueError(f'Only works in 2 and 3 dimension up to now!')

        # TODO rewrite function

        df_trajectories ={}
        for i in range(self.n_geodesics):
            df_trajectories[i] = pd.DataFrame(self.trajectories()[:,i,:],\
                                              columns=["x"+str(i) for i in range(self.dim)])


        if self.dim==2:
            fig = go.Figure()
            for i in range(self.n_geodesics):
                fig.add_trace(go.Scatter(x=df_trajectories[i]["x0"], y=df_trajectories[i]["x1"]))
            #fig.update_layout(width=500,
            #                    height=500)
            fig.update_yaxes(
                    scaleanchor = "x",
                    scaleratio = 1,
                  )
            fig.show()
        elif self.dim==3:
            length_trajectories = geodesics.trajectories().shape[0]
            df_trajectories = pd.DataFrame(np.einsum('ijk->jik',geodesics.trajectories()).reshape(-1,3), columns=["x"+str(i) for i in range(3)])
            df_trajectories['index'] = [int(i/length_trajectories) for i in range(df_trajectories.shape[0])]
            
            fig = px.line_3d(df_trajectories, x="x0", y="x1", z="x2", color='index')
            fig.show()
            
    def plot_endpoints(self, plot_file=None):
        try:
            assert(self.dim==3)
        except:
            raise ValueError(f'Only works in 3 dimension up to now!')
        
        if self.dim==3:
            df_endpoints = pd.DataFrame(self.endpoints(), columns=["x"+str(i) for i in range(3)])
            fig = px.scatter_3d(df_endpoints, x="x0", y="x1", z="x2")
            fig.show()
            if plot_file:
                fig.write_html('plots/' + plot_file)
                
    def plot_endpoints_with_dataset(self, df_dataset, plot_file: bool=None, filter_fkt=None, verbose=False):
        try:
            assert(self.dim==3)
        except:
            raise ValueError(f'Only works in 3 dimension up to now!')

        if self.dim==3:
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
                    fig.write_html(os.ṕath.join('plots', plot_file))
                except:
                    os.mkdir('plots')
                    fig.write_html(os.ṕath.join('plots', plot_file))
        #TODO
        #def plot_histogram_function_values(self,(a,b)):
        #    pass