import torch
from random import randint

def loss(output,target, index=-1, logfn=None):
    """
    output and target both lists of tensors length 4 such that:
    output[0] is vector magnetic field
    output[1] is gradBx
    output[2] is gradBy
    output[3] is gradBz
    target[0] is vector magnetic field
    target[1] is presx
    target[2] is presy
    target[3] is presz
    
    index is for a multiobjective method where only one dimension
    of the loss function is considered,rather than a sum-of-squares
    technique. -1 denotes this method is not being used, while
    0-6 denotes, in order, the three components of vector difference,
    mhs discrepancy, and divergence (all squared individually)
    """ 
    vec_diff = torch.sum(torch.square(output[0]-target[0]),0) /output[0].shape[0]
    mhs_diff = torch.sum(torch.square(torch.cat((
                          torch.unsqueeze((output[1][:,2]-output[3][:,0])*output[0][:,2]-
                                          (output[2][:,0]-output[1][:,1])*output[0][:,1] - target[1],1),
                          torch.unsqueeze((output[2][:,0]-output[1][:,1])*output[0][:,0]-
                                          (output[3][:,1]-output[2][:,2])*output[0][:,2] - target[2],1),
                          torch.unsqueeze((output[3][:,1]-output[2][:,2])*output[0][:,1]-
                                          (output[1][:,2]-output[3][:,0])*output[0][:,0] - target[3],1)
                        ),1)),0) /output[0].shape[0]
    div_diff = torch.sum(torch.square(torch.cat((
                          output[1][:,0],output[2][:,1],output[3][:,2]),0,))) /output[0].shape[0]
    
    loss = torch.cat((vec_diff,mhs_diff,torch.unsqueeze(div_diff,0)))
    vec_diff = torch.sum(vec_diff)
    mhs_diff = torch.sum(mhs_diff)
    div_diff = torch.sum(div_diff)
    if logfn is not None:
        logfn(f'--vec:{vec_diff}, mhs:{mhs_diff}, div:{div_diff}--')
    
    if index == 'random':
        r = randint(0,2)
        if r == 0:
            index = [0,1,2]
        elif r == 1:
            index = [3,4,5]
        else:# r == 2:
            index = -1
    
    if type(index) == int and (index < 0 or index > 6):
        index = [0,1,2,3,4,5,6]
        
    return torch.sum(loss[index]), vec_diff.item(), mhs_diff.item(), div_diff.item()
        
    