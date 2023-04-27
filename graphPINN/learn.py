import torch
import torch_geometric as pyg
import numpy as np
from time import time
from graphPINN.MHS import loss as MHSloss
from graphPINN import ddp, debug
import sys

def train(model, trainset, validset, epochs = 1, start_epoch = 0, logfn=None, optmethod = torch.optim.Adam, lossindex=-1, checkpointfile = '', use_ddp=False):
    if use_ddp:
        # Catch ddp and forward to the proper function
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        calltime = time()
        ddp.mp_train(n_gpus,model,trainset,validset,
                     epochs=epochs, start_epoch=start_epoch, optmethod=optmethod, lossindex=lossindex,
                     logfn=logfn,checkpointfile=f'{checkpointfile}_{calltime}')
        try:
            training_loss = torch.load(f'{checkpointfile}_{calltime}train.pt')
            validation_loss = torch.load(f'{checkpointfile}_{calltime}valid.pt')
            model = torch.load(f'{checkpointfile}_{calltime}model.pt')
        except FileNotFoundError:
            logfn('Checkpoint file not found.')
            raise RuntimeError('Multithread closed unexpectedly.')
    else:
        training_loss   = torch.zeros(4,epochs,len(lossindex))
        validation_loss = torch.zeros(4,epochs,len(lossindex))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        for index in range(len(lossindex)):
            logfn(f'-- starting index {index} = {lossindex[index]} --')
            
            for epoch in range(start_epoch, epochs):
                start_time = time()
                model.train(True)

                training_loss[:,epoch,index] = runEpoch(model, trainset, optmethod = optmethod,
                                                  logfn=logfn, lossindex=lossindex[index], use_tqdm=False
                                                 )
                logfn(f'Epoch {epoch+1} completed. Loss: {training_loss[3,epoch,index]}; Total time: {time()-start_time}')
                logfn(f'running vec: {training_loss[0,epoch,index]}, running mhs: {training_loss[1,epoch,index]}, running div: {training_loss[2,epoch,index]}')

                start_time = time()
                model.train(False)
                if len(checkpointfile) > 0:
                    torch.save(model.state_dict(), checkpointfile + f'epoch-{epoch+1}.pt')

                validation_loss[:,epoch,index] = runEpoch(model, validset, optmethod = None,
                                                  logfn=logfn, lossindex=lossindex[index], use_tqdm=False
                                                 )
                logfn(f'Validation loss: {validation_loss[3,epoch,index]}; validation time: {time()-start_time}')
                logfn(f'running vec: {validation_loss[0,epoch,index]}, running mhs: {validation_loss[1,epoch,index]}, running div: {validation_loss[2,epoch,index]}')
            if len(checkpointfile) > 0:
                torch.save(model.state_dict(),f'{checkpointfile}model-index{index}.pt')
            
    return training_loss, validation_loss, model.state_dict()

def runEpoch(model, dataset, optmethod = torch.optim.Adam, logfn=None, lossindex=-1, ddpRank = None, world_size=1, epoch=0, use_tqdm=True):
    if ddpRank is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        assert world_size>=2, 'Must set world_size >= 2 if using ddp.'
        device = ddpRank
    def closure():
        # necessary for lbfgs
        optimizer.zero_grad()
        output = model(data)
        loss,_,_,_ = MHSloss(output, true)
        loss.backward()
        return loss
    if model.training:
        optimizer = optmethod(model.parameters())
    
    if ddpRank is None:
        batch_size = 3
        loader = pyg.loader.DataListLoader(dataset, batch_size=batch_size,shuffle=False)
    else:
        batch_size = 6
        loader = ddp.DistributedLoader(ddpRank, world_size, dataset, batch_size=batch_size)
        loader.sampler.set_epoch(epoch)
        
    running_loss = 0
    running_vec = 0
    running_mhs = 0
    running_div = 0
    iter = 0
    skipped = 0
    
    if not use_tqdm:
        def tqdm(iterable):
            return iterable
    else:
        from tqdm.notebook import tqdm
        
    for data in tqdm(loader):
        
        if model.training:
            optimizer.zero_grad(set_to_none = True)
            
        for j in range(len(data)):
            data[j] = data[j].to_homogeneous().to(device)
        true = [torch.cat([datum.y[:,0:3] for datum in data]),
                torch.cat([datum.x[:,3] for datum in data]),
                torch.cat([datum.x[:,4] for datum in data]),
                torch.cat([datum.x[:,5] for datum in data])]
            
        pred = model.forward(data)
        loss, vec_diff, mhs_diff, div_diff = MHSloss(pred,true, index=lossindex, logfn=logfn)
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
        if ddpRank is None:
            logfn(f'  iter {iter}/{len(loader)}, loss {loss.item()}')
        else:
            logfn(f'  [{ddpRank}] iter {iter}/{len(loader)}, loss {loss.item()}')
    return torch.tensor((running_vec/len(loader),
                         running_mhs/len(loader),
                         running_div/len(loader),
                         running_loss/len(loader)
                       ))
