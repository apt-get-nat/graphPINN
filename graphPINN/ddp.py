import torch
import torch_geometric as pyg
import numpy as np
from time import time
import datetime
import os
import traceback

import torch.multiprocessing as mp
# import multiprocessing as mp
import torch.distributed as dist

import graphPINN

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=600))

def cleanup():
    dist.destroy_process_group()
    
def DistributedLoader(rank, world_size, dataset, batch_size=4):
    sampler = torch.utils.data.DistributedSampler(dataset,
                                                  num_replicas=world_size,rank=rank,
                                                  shuffle=False,drop_last=True
                                                 )
    dataloader = pyg.loader.DataListLoader(dataset,batch_size=batch_size,
                                shuffle=False,sampler=sampler,
                                num_workers=0,pin_memory=False,
                                drop_last=True
                               )
    return dataloader

def mp_train(world_size,model,trainset,validset, **kwargs):
    
    mp.spawn(train,
              args=(world_size,model,trainset,validset,kwargs),
              nprocs=world_size
             )

def train(rank, world_size, model, trainset, validset, *args):
    try:
        kwargs = args[0]
        epochs = kwargs.setdefault('epochs',1)
        start_epoch = kwargs.setdefault('start_epoch',0)
        optmethod = kwargs.setdefault('optmethod',0)
        lossindex = kwargs.setdefault('lossindex',[-1])
        logfn = kwargs.setdefault('logfn',None)
        checkpointfile = kwargs.setdefault('checkpointfile','')

        logfn(f'Starting on rank {rank}')
        setup(rank, world_size)

        device = rank % world_size
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device],output_device=device)

        training_loss   = torch.zeros(4,epochs,len(lossindex)).to(device)
        validation_loss = torch.zeros(4,epochs,len(lossindex)).to(device)
        for index in range(len(lossindex)):
            logfn(f'[{rank}] -- starting index {index} = {lossindex[index]} --')
            for epoch in range(start_epoch, epochs):
                start_time = time()
                model.train(True)
                training_loss[:,epoch,index] = graphPINN.learn.runEpoch(model, trainset, optmethod = optmethod, use_tqdm=False,
                                                  logfn=logfn, lossindex=lossindex[index], ddpRank=rank,world_size=world_size, epoch=epoch
                                                 )
                logfn(f'[{rank}] Epoch {epoch+1} completed. Loss: {training_loss[3,epoch,index]}; Total time: {time()-start_time}')
                logfn(f'[{rank}] running vec: {training_loss[0,epoch,index]}, running mhs: {training_loss[1,epoch,index]}, running div: {training_loss[2,epoch,index]}')

                model.train(False)
                if len(checkpointfile) > 0 and rank == 0:
                    torch.save(model.state_dict(), checkpointfile + f'epoch-{epoch+1}.pt')

                start_time = time()
                validation_loss[:,epoch,index] = graphPINN.learn.runEpoch(model, validset, optmethod = None, use_tqdm=False,
                                                    logfn=logfn, lossindex=lossindex[index], ddpRank=rank,world_size=world_size
                                                   )
                logfn(f'[{rank}] Validation loss: {validation_loss[3,epoch,index]}; validation time: {time()-start_time}')
                logfn(f'[{rank}] running vec: {validation_loss[0,epoch,index]}, running mhs: {validation_loss[1,epoch,index]}, running div: {validation_loss[2,epoch,index]}')

            torch.cuda.set_device(device)
            full_train_loss = [torch.Tensor(4,epochs,len(lossindex)).to(device) for _ in range(world_size)]
            full_valid_loss = [torch.Tensor(4,epochs,len(lossindex)).to(device) for _ in range(world_size)]
            dist.all_gather(full_train_loss, training_loss)
            dist.all_gather(full_valid_loss, validation_loss)
            if rank == 0:
                torch.save((sum(full_train_loss)/world_size).cpu(),f'{checkpointfile}train.pt')
                torch.save((sum(full_valid_loss)/world_size).cpu(),f'{checkpointfile}valid.pt')
                torch.save(model.module.module,f'{checkpointfile}model.pt')
                logfn(f'[{rank}] Model saved.')
        cleanup()
    except Exception as e:
        logfn(f'[{rank}] Error: {e}\n    {traceback.format_exc()}')
        cleanup()
    
