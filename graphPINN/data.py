import torch
import numpy as np
from scipy import io
import h5py
import torch_geometric as pyg
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import KNNGraph
import os.path as osp
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
        filename = 'D:\\nats ML stuff\\data\\raw\\sharp' + str(self.list_IDs[index]) + '.mat'
        try:
            mat = io.loadmat(filename)
        except NotImplementedError:
            mat = {}
            f = h5py.File(filename)
            for k,v in f.items():
                mat[k] = np.array(v)

        n = int(mat['n'])

        Bn = mat['Bns']
        Bn = np.stack((Bn[0:n,:],Bn[n:2*n,:],Bn[2*n:3*n,:]),0)
        Bn = np.transpose(Bn,(2,1,0))
        
        paramsn = np.transpose(mat['params'])

        nodesn = np.squeeze(mat['nodes'])
        nodesn = np.repeat(np.expand_dims(nodesn,0),6,axis=0)
        index_z0 = np.squeeze(mat['index_z0']).astype(int)
        Bn_bd = Bn[:,index_z0,:]
        nodesn_bd = nodesn[:,index_z0,:];
        
        if 'forcevec' in mat:
            # Should be standard for all data in the package, but wrapped in an if anyways
            # for backwards compatibility for how this data used to be presented (ie, only params)
            plasman = mat['forcevec']
            plasman = np.stack((plasman[0:n,:],plasman[n:2*n,:],plasman[2*n:3*n,:]),0)
            plasman = np.transpose(plasman,(2,1,0))
        else:
            plasman = None
        
        B = torch.Tensor(Bn[:,np.setdiff1d(range(n),index_z0),:])
        nodes = torch.Tensor(nodesn[:,np.setdiff1d(range(n),index_z0),:])
        B_bd = torch.Tensor(Bn_bd)
        nodes_bd = torch.Tensor(nodesn[:,index_z0,0:2])
        
        params = torch.Tensor(paramsn)
        
        if plasman is not None:
            plasma = torch.Tensor(plasman[:,np.setdiff1d(range(n),index_z0),:])
            plasma_bd = torch.Tensor(plasman[:,index_z0,:])
            return torch.utils.data.TensorDataset(nodes, B, nodes_bd, B_bd, params, plasma, plasma_bd)
        else:
            return torch.utils.data.TensorDataset(nodes, B, nodes_bd, B_bd, params)

class MHSDataset(pyg.data.Dataset):
    def __init__(self, root, k=50,transform=None, pre_transform=None, pre_filter=None):
        self.allSharps = _allsharps
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

                p_in = sim[5]
                p_bd = sim[6]
                
                data = pyg.data.HeteroData()
                data['in'].x = torch.cat((x_in,p_in),1)
                data['in'].y = y_in
                data['in'].pos = pos_in
                data['in','adj','in'].edge_index = KNNGraph(k=self.k)(data['in']).edge_index
                data['bd'].x = torch.cat((x_bd,p_bd),1)
                data['bd'].y = y_bd
                data['bd'].pos = pos_bd
                #data['in','propagates','bd'].edge_index = pyg.utils.dense_to_sparse(torch.ones(x_in.shape[0],x_bd.shape[0]))
                

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    self.pre_transform(data=data['in'])
                    
                torch.save(data, osp.join(self.processed_dir, f'simulation_{counter}.pt'))
                counter += 1
                
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, index):
        data = torch.load(osp.join(self.processed_dir, f'simulation_{index}.pt'))
        return data
    

_allsharps = [7058, 7066, 7067, 7069, 7070, 7074, 7075, 7078, 7081,
                          7083, 7084, 7085, 7088, 7090, 7096, 7097, 7099, 7100,
                          7103, 7107, 7109, 7110, 7112, 7115, 7116, 7117, 7118,
                          7120, 7121, 7122, 7123, 7127, 7128, 7130, 7131, 7134,
                          7139, 7140, 7141, 7144, 7147, 7148, 7153, 7161, 7163,
                          7164, 7165, 7166, 7167, 7169, 7170, 7171, 7172, 7182,
                          7183, 7188, 7189, 7190, 7192, 7193, 7194, 7195, 7201,
                          7203, 7204, 7205, 7209, 7211, 7212, 7214, 7215, 7220,
                          7221, 7222, 7224, 7227, 7228, 7229, 7230, 7235, 7236,
                          7237, 7240, 7241, 7242, 7245, 7246, 7248, 7251, 7254,
                          7255, 7256, 7259, 7260, 7261, 7262, 7267, 7269, 7274,
                          7275, 7276, 7277, 7283, 7287, 7290, 7292, 7293, 7298,
                          7299, 7300, 7301, 7302, 7304, 7305, 7307, 7308, 7310,
                          7312, 7313, 7316, 7319, 7322, 7323, 7324, 7325, 7326,
                          7327, 7328, 7329, 7331, 7334, 7335, 7336, 7337, 7341,
                          7346, 7348, 7350, 7351, 7352, 7353, 7356, 7357, 7358,
                          7359, 7363, 7366, 7368, 7369, 7372, 7373, 7374, 7375,
                          7376, 7379, 7380, 7382, 7384, 7385, 7386, 7389, 7398,
                          7399, 7400, 7401, 7402, 7403, 7405, 7406, 7410, 7412,
                          7413, 7415, 7418, 7419, 7420, 7422, 7424, 7425, 7430,
                          7431, 7432, 7435, 7436, 7438, 7439, 7440, 7442, 7443,
                          7450, 7451, 7452, 7453, 7454, 7458, 7459, 7461, 7463,
                          7464, 7466, 7468, 7469, 7470, 7471, 7472, 7476, 7477,
                          7484, 7487, 7490, 7493, 7498, 7499, 7502, 7503, 7504,
                          7507, 7508, 7509, 7510, 7511, 7513, 7518, 7521, 7525,
                          7529, 7530, 7532, 7534, 7535, 7536, 7538, 7540, 7541,
                          7542, 7543, 7544, 7545, 7546, 7547, 7548, 7549, 7550,
                          7551, 7552, 7553, 7554, 7555, 7558, 7561, 7562, 7563,
                          7564, 7566, 7568, 7569, 7570, 7571, 7572, 7573, 7575,
                          7580, 7581, 7583, 7585, 7589, 7590, 7592, 7594, 7595,
                          7596, 7598, 7599, 7600, 7602, 7604, 7605, 7607, 7608,
                          7610, 7612, 7613, 7614, 7617, 7618, 7619, 7623, 7624,
                          7628, 7630, 7633, 7634, 7635, 7638, 7640, 7643, 7645,
                          7648, 7651, 7652, 7653, 7655, 7658, 7659, 7660, 7661,
                          7662, 7665, 7666, 7667, 7670, 7672, 7673, 7674, 7675,
                          7679, 7681, 7685, 7689, 7690, 7691, 7693, 7694, 7698,
                          7699, 7703, 7706, 7708, 7709, 7710, 7711, 7713, 7716,
                          7718, 7719, 7720, 7721, 7723, 7724, 7725, 7726, 7728,
                          7730, 7731, 7733, 7737, 7739, 7740, 7741, 7744, 7745,
                          7746, 7747, 7748, 7749, 7751, 7752, 7753, 7754, 7755,
                          7759, 7760, 7762, 7764, 7765, 7766, 7770, 7771, 7773,
                          7777, 7778, 7779, 7780, 7781, 7785, 7786, 7787, 7788,
                          7789, 7790, 7791, 7793, 7798, 7799, 7803, 7805, 7807,
                          7808, 7813, 7814, 7818, 7819, 7821, 7825, 7827, 7833,
                          7835, 7838, 7840, 7842, 7845, 7847, 7848, 7849, 7850,
                          7852, 7853, 7854, 7855, 7856, 7857, 7860, 7861, 7862,
                          7863, 7870, 7871, 7872, 7873, 7878, 7881, 7882, 7883,
                          7884, 7886, 7888, 7890, 7891, 7896, 7897, 7898, 7901,
                          7905, 7906, 7911, 7912, 7913, 7917, 7918, 7919, 7921,
                          7922, 7923, 7924, 7929, 7932, 7933, 7934, 7936, 7937,
                          7939, 7942, 7944, 7947, 7950, 7951, 7952, 7953, 7959,
                          7961, 7966, 7967, 7969, 7973, 7975, 7976, 7978, 7981,
                          7982, 7983, 7985, 7989, 7991, 7992, 7994, 7997, 7999,
                          8000, 8002, 8003, 8005, 8006, 8009, 8010, 8012, 8013,
                          8014, 8016, 8017, 8020, 8022, 8023, 8025, 8026, 8027,
                          8028, 8029, 8030, 8031, 8032, 8033, 8038, 8041, 8043,
                          8049, 8050, 8051, 8052, 8057, 8060, 8064, 8067, 8070,
                          8071, 8072, 8075, 8077, 8078, 8081, 8082, 8083, 8086,
                          8088, 8089, 8092, 8093, 8094, 8095, 8096, 8097, 8098,
                          8102, 8104, 8105, 8109, 8113, 8114, 8115, 8116, 8123,
                          8129, 8132, 8133, 8134, 8137, 8138, 8139, 8140, 8141,
                          8142, 8143, 8148, 8149, 8150, 8151, 8155, 8159, 8164,
                          8166, 8167, 8171, 8172, 8174, 8175, 8180, 8181, 8182,
                          8185, 8188, 8195, 8197, 8198, 8204, 8205, 8206, 8207,
                          8212, 8213, 8214, 8217, 8219, 8221, 8225, 8226, 8227,
                          8228, 8229, 8230, 8239, 8240, 8241, 8242, 8245, 8247,
                          8248, 8250, 8251, 8252, 8253, 8256, 8257, 8259, 8260,
                          8263, 8264, 8266, 8273, 8275, 8278, 8281, 8282, 8285,
                          8286, 8288, 8290, 8292, 8294, 8298, 8300, 8301, 8319,
                          8325, 8326, 8330, 8334, 8335, 8343, 8346, 8347, 8348,
                          8349, 8350, 8351, 8353, 8356, 8358, 8359, 8362, 8364,
                          8365, 8366, 8367, 8368, 8369]
    
    