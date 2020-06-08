import time
import torch
import shutil
import os

import numpy as np

from tqdm import tqdm

from external.python_plyfile.plyfile import PlyElement, PlyData
from PCLosses import acc_cov

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def trainer(train_loader, val_loader, test_loader, model, optimizer, scheduler, criterion, epochs=10, export=None, infer=None, data_representation="pointcloud"):
    if export is None:
        export = "../model/dummy_model"

    best_loss = 9e99
    for epoch in range(epochs):

        # train for one epoch
        loss_training = train(train_loader, model, criterion, optimizer, epoch, data_representation=data_representation)

        # evaluate on validation set
        loss_validation = validate(val_loader, model, criterion, epoch, data_representation=data_representation)

        # update the LR scheduler
        if scheduler is not None:
            scheduler.step(loss_validation)

        # remember best prec@1 and save checkpoint
        is_best = loss_validation < best_loss
        best_loss = max(loss_validation, best_loss)
        
        
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(state, export+data_representation+"_epoch"+str(epoch)+ "_bneck" + str(model.bneck_size)+ ".pth.tar")
        if is_best:
            torch.save(state, export+data_representation + "_bneck" + str(model.bneck_size)+"_best.pth.tar")
            if infer is not None:
                test(test_loader, model, epoch, save=infer, data_representation=data_representation)

                
                
                
                
def train(train_loader, model, criterion, optimizer, epoch, data_representation="pointcloud"):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    avg_cov = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    with tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)  as t:
        for i, (sample, target) in t:
            # measure data loading time
            data_time.update(time.time() - end)

            sample = sample.cuda()
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(sample)
            loss = criterion(output, target)
            if(data_representation=="pointcloud"):
                acc, cov = acc_cov(output,target)

            # measure accuracy and record loss
            losses.update(loss.item(), sample.size(0))
            if(data_representation=="pointcloud"):
                avg_acc.update(torch.mean(acc), sample.size(0))
                avg_cov.update(torch.mean(cov), sample.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if(data_representation=="pointcloud"):
                t.set_description('Training {0}: '
                              'Time {batch_time.avg:.3f}s (it:{batch_time.val:.3f}s, data:{data_time.val:.3f}s) '
                              'Acc {acc.avg:.4f} (it:{acc.val:.4f}) Cov {cov.avg:.4f} (it:{cov.val:.4f}) Loss {loss.avg:.4f} (it:{loss.val:.4f})'.format(
                               epoch, batch_time=batch_time, data_time=data_time, loss=losses, acc=avg_acc, cov = avg_cov))
            elif(data_representation=="voxel"):
                t.set_description('Training {0}: '
                              'Time {batch_time.avg:.3f}s (it:{batch_time.val:.3f}s, data:{data_time.val:.3f}s) '
                              'Loss {loss.avg:.4f} (it:{loss.val:.4f})'.format(
                               epoch, batch_time=batch_time, data_time=data_time, loss=losses))


    return losses.avg
                
          
        
        
        
        
def validate(val_loader, model, criterion, epoch, data_representation="pointcloud"):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    avg_cov = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with tqdm(enumerate(val_loader), total=len(val_loader), ncols=120)  as t:

        with torch.no_grad():
            end = time.time()
            for i, (sample, target) in t:
                # measure data loading time
                data_time.update(time.time() - end)
                
                sample = sample.cuda()
                target = target.cuda(non_blocking=True)

                # compute output
                output = model(sample)
                loss = criterion(output, target)
                if(data_representation=="pointcloud"):
                    acc, cov = acc_cov(output,target)

                # measure accuracy and record loss
                losses.update(loss.item(), sample.size(0))
                if(data_representation=="pointcloud"):
                    avg_acc.update(torch.mean(acc), sample.size(0))
                    avg_cov.update(torch.mean(cov), sample.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()


                if(data_representation=="pointcloud"):
                    t.set_description('Validation {0}: '
                              'Time {batch_time.avg:.3f}s (it:{batch_time.val:.3f}s, data:{data_time.val:.3f}s) '
                              'Acc {acc.avg:.4f} (it:{acc.val:.4f}) Cov {cov.avg:.4f} (it:{cov.val:.4f}) Loss {loss.avg:.4f} (it:{loss.val:.4f})'.format(
                               epoch, batch_time=batch_time, data_time=data_time, loss=losses, acc=avg_acc, cov = avg_cov))
                elif(data_representation=="voxel"):
                    t.set_description('Validation {0}: '
                              'Time {batch_time.avg:.3f}s (it:{batch_time.val:.3f}s, data:{data_time.val:.3f}s) '
                              'Loss {loss.avg:.4f} (it:{loss.val:.4f})'.format(
                               epoch, batch_time=batch_time, data_time=data_time, loss=losses))


    return losses.avg


def write_ply_tensor(tensor, filename):
    tensor = tensor.t().cpu().numpy()
    tensor = [tuple(element) for element in tensor]
    el = PlyElement.describe(np.array(tensor, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
    PlyData([el]).write(filename)



def test(test_loader, model, epoch, save=None, data_representation="pointcloud"):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    #Prepare directories for save
    if save is not None:
        in_dir = save + "/in/"
        if not os.path.exists(os.path.dirname(in_dir)):
            os.makedirs(os.path.dirname(in_dir))
        gt_dir = save + "/gt/"
        if not os.path.exists(os.path.dirname(gt_dir)):
            os.makedirs(os.path.dirname(gt_dir))
        out_dir = save + '/' + str(model.bneck_size) + '/'
        if not os.path.exists(os.path.dirname(out_dir)):
            os.makedirs(os.path.dirname(out_dir))

    # switch to evaluate mode
    model.eval()

    with tqdm(enumerate(test_loader), total=len(test_loader), ncols=120)  as t:
        with torch.no_grad():
            end = time.time()
            for i, (sample, target) in t:            
                # measure data loading time
                data_time.update(time.time() - end)
                
                sample = sample.cuda()
                # compute output
                output = model(sample)

                #Save first batch only
                if save is not None and i==0:
                    if(data_representation=="pointcloud"):
                        # save point clouds
                        for j, cur_target in enumerate(target):
                            write_ply_tensor(cur_target, gt_dir + str(j) + '.ply')
                            write_ply_tensor(sample[j], in_dir + str(j) + '.ply')
                            write_ply_tensor(output[j], out_dir + str(j) +'.ply')

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                    
                t.set_description('Test {0}: '
                                  'Time {batch_time.avg:.3f}s (it:{batch_time.val:.3f}s, data:{data_time.val:.3f}s) '.format(
                                  epoch, batch_time=batch_time, data_time=data_time))
               

    return 0