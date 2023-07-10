import torch
import numpy as np
from scipy import io
import h5py
import torch_geometric as pyg
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import KNNGraph, RadiusGraph
import os
from tqdm import tqdm

class SHARPData(torch.utils.data.Dataset):
    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    def __getitem__(self, index):
        'Generates one sample of data'
        filename = _rawfolder + 'sharp' + str(self.list_IDs[index]) + '.mat'
        try:
            mat = io.loadmat(filename)
        except NotImplementedError:
            mat = {}
            f = h5py.File(filename)
            for k,v in f.items():
                mat[k] = np.array(v)

        n = int(mat['n'])

        Bn = np.concatenate((mat['Bns'],mat['Bff']),1)
        Bn = np.stack((Bn[0:n,:],Bn[n:2*n,:],Bn[2*n:3*n,:]),0)
        Bn = np.transpose(Bn,(2,1,0))
        

        nodesn = np.squeeze(mat['nodes'])
        nodesn = np.repeat(np.expand_dims(nodesn,0),Bn.shape[0],axis=0)
        index_z0 = np.squeeze(mat['index_z0']).astype(int)
        Bn_bd = Bn[:,index_z0,:]
        nodesn_bd = nodesn[:,index_z0,:];
        
        plasman = np.concatenate((np.zeros((3*n,1)),mat['forcevec']),1)
        plasman = np.stack((plasman[0:n,:],plasman[n:2*n,:],plasman[2*n:3*n,:]),0)
        plasman = np.transpose(plasman,(2,1,0))
        
        
        B = torch.Tensor(Bn[:,np.setdiff1d(range(n),index_z0),:])
        nodes = torch.Tensor(nodesn[:,np.setdiff1d(range(n),index_z0),:])
        B_bd = torch.Tensor(Bn_bd)
        nodes_bd = torch.Tensor(nodesn[:,index_z0,0:2])
        
        plasma = torch.Tensor(plasman[:,np.setdiff1d(range(n),index_z0),:])
        plasma_bd = torch.Tensor(plasman[:,index_z0,:])
        
        sharp = torch.full((Bn.shape[0],1), self.list_IDs[index])
        
        return torch.utils.data.TensorDataset(nodes, B, nodes_bd, B_bd, plasma, plasma_bd, sharp)

class MHSDataset(pyg.data.Dataset):
    def __init__(self, root, k=50,transform=None, pre_transform=None, pre_filter=None):
        self.allSharps = get_allsharps()
        self.k=k
        super().__init__(root, transform, pre_transform, pre_filter)
    @property
    def raw_file_names(self):
        return ['sharp' + str(s) + '.mat' for s in self.allSharps]
    
    @property
    def processed_file_names(self):
        return ['simulation_' + str(t) + '.pt' for t in range(6 * len(self.allSharps))]
    
    def process(self):
        
        tensorData = SHARPData(self.allSharps)
        
        numSharps = len(tensorData)
        numPerSharp = len(tensorData[0])
        
        counter = 0
        for sharp_set in tqdm(tensorData):
            for sim in sharp_set:
                
                x_in = torch.zeros(sim[1].shape)
                x_bd = sim[3]
                y_in = sim[1]
                y_bd = sim[3]
                pos_in = sim[0]
                pos_bd = torch.cat((sim[2],torch.zeros(sim[2].shape[0],1)),1)

                p_in = sim[4]
                p_bd = sim[5]
                
                
                data = pyg.data.HeteroData()
                data['in'].x = torch.cat((x_in,p_in),1)
                data['in'].y = y_in
                data['in'].pos = pos_in
                data['in','adj','in'].edge_index = KNNGraph(k=self.k)(data['in']).edge_index
                data['in'].edge_index = None
                data['bd'].x = torch.cat((x_bd,p_bd),1)
                data['bd'].y = y_bd
                data['bd'].pos = pos_bd
                
                data['bd','propagates','in'].edge_index, _ = \
                        pyg.utils.dense_to_sparse(
                                torch.ones(data['bd'].x.shape[0],data['in'].x.shape[0])
                        )
                data['bd','propagates','in'].edge_index, mask = pyg.utils.dropout_edge(
                        edge_index = data['bd','propagates','in'].edge_index, p = 0.8,
                        training=True
                )
#                 data['bd','propagates','in'].edge_attr = data['bd','propagates','in'].edge_attr[mask]

#                 data['bd','propagates','in'].edge_index = RadiusGraph()(data)
                
                
                data.sharpnum = sim[6]
                

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    self.pre_transform(data=data['in'])
                    
                torch.save(data, os.path.join(self.processed_dir, f'simulation_{counter}.pt'))
                counter += 1
                
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, index):
        data = torch.load(os.path.join(self.processed_dir, f'simulation_{index}.pt'))
        return data
    
def get_allsharps():
    # return _allsharps
    sharplist = os.listdir(_rawfolder)
    allsharps = []
    for filename in sharplist:
        allsharps.append(int(filename.replace('sharp','').replace('.mat','')))
    return allsharps
    
_rawfolder = 'D:\\MHS_solutions_v4\\'
_allsharps = [7058,7066,7067,7069,7070,7074,7078,7081,7083,7084,7085]
    