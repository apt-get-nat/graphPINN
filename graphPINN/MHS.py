import torch

def loss(output,target, logfn=None):
    # output and target both lists of tensors length 4 such that:
    # output[0] is vector magnetic field
    # output[1] is gradBx
    # output[2] is gradBy
    # output[3] is gradBz
    # target[0] is vector magnetic field
    # target[1] is presx
    # target[2] is presy
    # target[3] is presz
    
    vec_diff = torch.sum(torch.square(output[0]-target[0])) /output[0].shape[0]*1e2
    mhs_diff = torch.sum(torch.square(torch.cat((
                          (output[1][:,2]-output[3][:,0])*output[0][:,2]-(output[2][:,0]-output[1][:,1])*output[0][:,1] - target[1],
                          (output[2][:,0]-output[1][:,1])*output[0][:,0]-(output[3][:,1]-output[2][:,2])*output[0][:,2] - target[2],
                          (output[3][:,1]-output[2][:,2])*output[0][:,1]-(output[1][:,2]-output[3][:,0])*output[0][:,0] - target[3]
                        ),0))) /output[0].shape[0] * 1e5
    div_diff = torch.sum(torch.square(torch.cat((
                          output[1][:,0],output[2][:,1],output[3][:,2]),0,))) /output[0].shape[0] * 1e4
    if logfn is not None:
        logfn(f'--vec:{vec_diff}, mhs:{mhs_diff}, div:{div_diff}--')
    
    return vec_diff + mhs_diff + div_diff, vec_diff.item(), mhs_diff.item(), div_diff.item()