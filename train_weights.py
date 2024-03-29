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
    
    folder = "C:\\Users\\NASA\\Documents\\ML_checkpoints\\2023-04-27"
    if not os.path.exists(folder):
        os.makedirs(folder)
    logfn = graphPINN.debug.Logfn(folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for j in range(torch.cuda.device_count()):
        logfn(f"{j}: {graphPINN.debug.pretty_size(torch.cuda.get_device_properties(j).total_memory)} of {'cuda' if torch.cuda.is_available() else 'cpu'} memory")
    
    k = 100
    ddp = False

    dataset = graphPINN.data.MHSDataset(f'D:\\scattered_data_k={k}',k=k)
    propdesign = [12,6,3]
    # propdesign = [12,3]
    convdesign = [18,9,6,3]
    # convdesign = [18,3]

    propkernel = graphPINN.KernelNN(propdesign, torch.nn.ReLU)
    propgraph = graphPINN.BDPropGraph(propkernel)
    convkernel = graphPINN.KernelNN(convdesign, torch.nn.ReLU)
    convgraph = graphPINN.ConvGraph(convkernel)
#     model = pyg.nn.DataParallel(graphPINN.FullModel(propgraph, convgraph))
    model = graphPINN.FullModel(propgraph, convgraph)

    trainset, validset, testset = torch.utils.data.random_split(dataset,[0.8, 0.1, 0.1],generator=torch.Generator().manual_seed(314))
#     trainset, validset, testset = torch.utils.data.random_split(dataset,[0.01, 0.005, 0.985],generator=torch.Generator().manual_seed(314))
    
    lossindex = [[0,1,2],-1]
    epochs = 10
    training_loss, validation_loss, state_dict = graphPINN.learn.train(
                model, trainset, validset, lossindex=lossindex,
                epochs=epochs, logfn=logfn, checkpointfile=folder, use_ddp = ddp)
    model.load_state_dict(state_dict)
    lossdict = {}
    lossdict[f'trainloss'] = training_loss.numpy()
    lossdict[f'validloss'] = validation_loss.numpy()
    logfn(f'training loss:\n{lossdict[f"trainloss"]}')
    logfn(f'validation loss:\n{lossdict[f"validloss"]}')
    
    lossdict['index_array'] = str(lossindex)
    torch.save(model, f'{folder}model_trainsize-{len(trainset)}_k-{k}_params-{math.prod(convdesign)+math.prod(propdesign)}.pt')
    savemat(f'{folder}loss_{epochs}_trainsize-{len(trainset)}_k-{k}_params-{math.prod(convdesign)+math.prod(propdesign)}.mat',lossdict)
    
    
if __name__ == "__main__":
#     start_monitoring(seconds_frozen=400)
    main()