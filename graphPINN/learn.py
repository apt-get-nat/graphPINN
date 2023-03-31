import torch
import torch_geometric as pyg
import numpy as np
from tqdm.notebook import tqdm
from time import time
from graphPINN.MHS import loss as MHSloss

def train(model, trainset, validset, epochs = 1, start_epoch = 0, logfn=None, optmethod = torch.optim.Adam, lossindex=-1, ddp = False, checkpointfile = ''):
    training_loss   = torch.zeros(4,epochs)
    validation_loss = torch.zeros(4,epochs)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if ddp:
        model = pyg.nn.DataParallel(model.to(device))
    else:
        model = model.to(device)
    
    for epoch in range(start_epoch, epochs):
        start_time = time()
        model.train(True)
        
        training_loss[:,epoch] = runEpoch(model, trainset, optmethod = optmethod,
                                          logfn=logfn, lossindex=lossindex, ddp = ddp
                                         )
        logfn(f'Epoch {epoch+1} completed. Loss: {training_loss[3,epoch]}; Total time: {time()-start_time}')
        logfn(f'running vec: {training_loss[0,epoch]}, running mhs: {training_loss[1,epoch]}, running div: {training_loss[2,epoch]}')
        
        model.train(False)
        if len(checkpointfile) > 0:
            torch.save(model.state_dict(), checkpointfile + f'epoch-{epoch+1}.pt')
        
        validation_loss[:,epoch] = runEpoch(model, validset, optmethod = None,
                                          logfn=logfn, lossindex=lossindex, ddp = ddp
                                         )
        logfn(f'Validation loss: {validation_loss[3,epoch]}; validation time: {time()-start_time}')
        logfn(f'running vec: {validation_loss[0,epoch]}, running mhs: {validation_loss[1,epoch]}, running div: {validation_loss[2,epoch]}')
        
    return training_loss, validation_loss

def runEpoch(model, dataset, optmethod = torch.optim.Adam, logfn=None, lossindex=-1, ddp = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def closure():
        # necessary for lbfgs
        optimizer.zero_grad()
        output = model(data)
        loss,_,_,_ = MHSloss(output, true)
        loss.backward()
        return loss
    if model.training:
        optimizer = optmethod(model.parameters())
    
    if ddp:
        batch_size = 6
        loader = pyg.loader.DataListLoader(dataset, batch_size=batch_size,shuffle=False)
    else:
        batch_size = 1
        loader = pyg.loader.DataLoader(dataset, batch_size=batch_size,shuffle=False)
        
    running_loss = 0
    running_vec = 0
    running_mhs = 0
    running_div = 0
    iter = 0
    skipped = 0

    for data in tqdm(loader):
        if model.training:
            optimizer.zero_grad()
            
        if ddp:
            for j in range(len(data)):
                data[j] = data[j].to(device).to_homogeneous()
            true = [torch.cat([datum.y[:,0:3] for datum in data]),
                    torch.cat([datum.x[:,3] for datum in data]),
                    torch.cat([datum.x[:,4] for datum in data]),
                    torch.cat([datum.x[:,5] for datum in data])]
        else:
            data = data.to(device).to_homogeneous()
            true = [data.y[:,0:3],data.x[:,3],data.x[:,4],data.x[:,5]]
            
        pred = model(data)

        loss, vec_diff, mhs_diff, div_diff = MHSloss(pred,true, index=lossindex, logfn=None)

        iter += 1

        if model.training:
            loss.backward()
            try:
                optimizer.step()
            except TypeError:
                optimizer.step(closure)

        running_loss += loss.item()
        running_vec += vec_diff
        running_mhs += mhs_diff
        running_div += div_diff
        logfn(f'  iter {iter}/{len(loader)}, loss {loss.item()}')
        
    return torch.tensor((running_vec/len(loader),
                         running_mhs/len(loader),
                         running_div/len(loader),
                         running_loss/len(loader)
                       ))
