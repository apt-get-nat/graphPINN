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
        # kernel a torch module with input size 12 and output size 3
        super().__init__(aggr='mean')
        self.kernel = kernel
    
    def reset_parameters(self):
        self.kernel.reset_parameters()
    
    def message(self,x_i,x_j, pos_i,pos_j,edge_attr):
        update = self.kernel.forward(torch.cat((pos_i,x_i,pos_j,x_j,edge_attr),1))
        
        return update
    
    def update(self,aggr_out):
        return aggr_out
    
    def forward(self,kdtree,iter=1):
        gradBx,gradBy,gradBz = None,None,None
        
        positions = [kdtree.pos[j,:].unsqueeze(0).requires_grad_() for j in range(kdtree.pos.shape[0])]
        x = kdtree.x
        for _ in range(iter):
            x = self.propagate(kdtree.edge_index, x=x, pos=torch.cat(positions,0), edge_attr=kdtree.edge_attr)
        
        gradBx = torch.autograd.grad([x[j,0] for j in range(x.shape[0])], positions, retain_graph=True)
        gradBy = torch.autograd.grad([x[j,1] for j in range(x.shape[0])], positions, retain_graph=True)
        gradBz = torch.autograd.grad([x[j,2] for j in range(x.shape[0])], positions, retain_graph=True)
        
        return [x, torch.cat(gradBx,0), torch.cat(gradBy,0), torch.cat(gradBz,0)]