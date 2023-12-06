import torch
import torch_geometric as pyg
import numpy as np

class KernelNN(torch.nn.Module):
    def __init__(self, layers, nonlinearity):
        super(KernelNN, self).__init__()
        
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = torch.nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(torch.nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class ConvGraph(pyg.nn.MessagePassing):
    def __init__(self, kernel):
        # kernel a torch module with input size 18 and output size 3
        super().__init__(aggr='mean')
        self.kernel = kernel
    
    def reset_parameters(self):
        self.kernel.reset_parameters()
    
    def message(self,x_i,x_j, pos_i,pos_j):
        update = self.kernel.forward(torch.cat((pos_i,x_i,pos_j,x_j),1))
        
        return update
    
    def update(self,aggr_out):
        return aggr_out
    
    def forward(self,tree,positions):
        gradBx,gradBy,gradBz = None,None,None
        
        y = self.propagate(tree['in','adj','in'].edge_index,
                           x=tree['in'].x, pos=torch.cat(positions,0))
        
        
        return y
   
class BDPropGraph(pyg.nn.MessagePassing):
    def __init__(self, kernel):
        super().__init__(aggr='mean')
        self.kernel = kernel
    def reset_parameters(self):
        self.kernel.reset_parameters()
    def message(self, x_i, x_j, pos_i, pos_j):
        update = self.kernel.forward(torch.cat((pos_i,x_i[:,3:6],x_j,pos_j),1))
        
        return update
    def update(self, aggr_out):
        return aggr_out
    
    def forward(self, tree,positions):
        y = self.propagate(
                    tree['bd','propagates','in'].edge_index, size=(tree['bd'].num_nodes,tree['in'].num_nodes),
                    x=(tree['bd'].x,tree['in'].x), pos=(tree['bd'].pos,torch.cat(positions,0))
        )
        return y
        

class FullModel(torch.nn.Module):
    def __init__(self,propgraph, convgraph, device='cpu'):
        super(FullModel, self).__init__()
        self.propgraph = propgraph
        self.convgraph = convgraph
        
    def forward(self,data):
        
        if isinstance(data, list):
            data = pyg.data.Batch.from_data_list(data)
        
        positions = [data['in'].pos[j,:].unsqueeze(0).requires_grad_() for j in range(data['in'].pos.shape[0])]
        
        data['in'].x[:,0:3] = self.propgraph.forward(data,positions)
        
        
        B = self.convgraph.forward(data, positions)

        gradBx = torch.autograd.grad([B[j,0] for j in range(B.shape[0])], positions, retain_graph=True)
        gradBy = torch.autograd.grad([B[j,1] for j in range(B.shape[0])], positions, retain_graph=True)
        gradBz = torch.autograd.grad([B[j,2] for j in range(B.shape[0])], positions, retain_graph=True)

        return [B, torch.cat(gradBx,0), torch.cat(gradBy,0), torch.cat(gradBz,0)]
    
