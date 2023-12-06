import numpy as np
import torch
import torch_geometric as pyg
import graphPINN
import math

from time import time
from scipy.io import savemat, loadmat
import os

from hanging_threads import start_monitoring

def main():
    # os.environ['MKL_THREADING_LAYER'] = 'GNU' # fixes a weird intel multiprocessing error with numpy
    
    folder = "C:\\Users\\NASA\\Documents\\ML_checkpoints\\2023-08-22\\"
    if not os.path.exists(folder):
        os.makedirs(folder)
    logfn = graphPINN.debug.Logfn(folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for j in range(torch.cuda.device_count()):
        logfn(f"{j}: {graphPINN.debug.pretty_size(torch.cuda.get_device_properties(j).total_memory)} of {'cuda' if torch.cuda.is_available() else 'cpu'} memory")
    
    k = 100
    ddp = True

    l = 100
    bd = 200
    dataset = graphPINN.data.MHSDataset(f'D:\\v4_set_k={k}_l={l}_bd={bd}',k=k, l=l, bd=bd)
    # layer 1 of prop: B_0 + x,y_0 + F_0 + x,y,z_node + F_node = 14
#     propdesign = [12,6,3]
    propdesign = [14,6,3]
    # layer 1 of conv: P_k + x,y,z_k + F_k + P_node + x,y,z_node + F_node = 18
#     convdesign = [18,9,6,3]
    convdesign = [18,12,3]

    propkernel = graphPINN.KernelNN(propdesign, torch.nn.ReLU)
    propgraph = graphPINN.BDPropGraph(propkernel)
    convkernel = graphPINN.KernelNN(convdesign, torch.nn.ReLU)
    convgraph = graphPINN.ConvGraph(convkernel)
    model = graphPINN.FullModel(propgraph, convgraph)

    trainset, validset, testset = torch.utils.data.random_split(dataset,[0.8, 0.1, 0.1],generator=torch.Generator().manual_seed(314))
#     trainset, validset, testset = torch.utils.data.random_split(dataset,[0.01, 0.005, 0.985],generator=torch.Generator().manual_seed(314))
    
    lossindex = [0,1,2,[3,4,5],[0,1,2]]
    epochs = 3
    
#    model = pyg.nn.DataParallel(graphPINN.FullModel(propgraph, convgraph))
#    model.load_state_dict(torch.load("C:\\Users\\NASA\\Documents\\ML_checkpoints\\2023-07-31\\_1691861661.7859707epoch-1.pt"))
#    model = model.module
    
    training_loss, validation_loss, state_dict = graphPINN.learn.train(
                model, trainset, validset, lossindex=lossindex,
                epochs=epochs, logfn=logfn, checkpointfile=folder, use_ddp = ddp)
    model.load_state_dict(state_dict)
    lossdict = { 'trainloss':  training_loss.numpy(),
                 'validloss':validation_loss.numpy(),
                 'index_array':str(lossindex)
               }
    logfn(f'training loss:\n{lossdict[f"trainloss"]}')
    logfn(f'validation loss:\n{lossdict[f"validloss"]}')
    
    torch.save(model, f'{folder}model_trainsize-{len(trainset)}_k-{k}_params-{math.prod(convdesign)+math.prod(propdesign)}.pt')
    savemat(f'{folder}loss_{epochs}_trainsize-{len(trainset)}_k-{k}_params-{math.prod(convdesign)+math.prod(propdesign)}.mat',lossdict)
    
    
if __name__ == "__main__":
#     start_monitoring(seconds_frozen=400)
    main()