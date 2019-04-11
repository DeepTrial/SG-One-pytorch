import os,cv2
import torch
import json
import argparse
from config import cfg
from OneShotModel import *
from Utils.LoadDataSeg import data_loader
from Utils.Restore import restore
from Utils import averageMeter
from Utils.ParaNumber import get_model_para_number
from Utils.util import save_checkpoint,get_save_dir
import numpy as np

ROOT_DIR = '/'.join(os.getcwd().split('/'))
SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots')



def get_arguments():
    parser = argparse.ArgumentParser(description='OneShotSegmentation')
    parser.add_argument("--arch", type=str,default='sg_one')
    parser.add_argument("--max_steps", type=int, default=cfg.train.max_steps)
    parser.add_argument("--lr", type=float, default=cfg.train.LR)
    parser.add_argument("--disp_interval", type=int, default=cfg.train.disp_interval)
    parser.add_argument("--save_interval", type=int, default=cfg.train.save_interval)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--batch_size", type=int, default=cfg.train.batch_size)
    parser.add_argument("--start_count", type=int, default=0)
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--group", type=int, default=cfg.train.group)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument("--dataset_name", type=str, default=cfg.DataSet_Name)
    return parser.parse_args()



def get_model(args):

    model = eval(args.arch).OneModel(args)
    model = model.cuda()

    print('Number of Parameters: %d'%(get_model_para_number(model)))
    # optimizer
    model_optimizer = optimizer.get_finetune_optimizer(args, model)

    snapshot_dir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d' % (args.group, args.num_folds))
    if args.resume:
        restore(snapshot_dir, model)
        print("Resume training...")

    return model,  model_optimizer




def train(args):

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))

    losses = averageMeter()
    model, optimizer= get_model(args)
    model.train()


    train_loader = data_loader(args)

    save_log_dir = get_save_dir(args)
    log_file  = open(os.path.join(save_log_dir, 'log_history_loss.txt'),'w')


    count = args.start_count
    for dataPacket in train_loader:
        if count > args.max_steps:
            break

        anchor_img, anchor_mask, pos_img, pos_mask, neg_img, neg_mask = dataPacket
        
        #neg_img_ca=pos_img.data.numpy()[0]
        #query_img=np.transpose(neg_img_ca,(1,2,0))
        #neg_mask_ca=pos_mask.data.numpy()[0]
        #cv2.imwrite('./snapshots/neg_mask'+str(count)+'.jpg',(neg_mask_ca*255.0).astype(np.int32))
        #cv2.imwrite('./snapshots/neg_img'+str(count)+'.jpg',(query_img*255.0).astype(np.int32))
        anchor_img, anchor_mask, pos_img, pos_mask, \
        = anchor_img.cuda(), anchor_mask.cuda(), pos_img.cuda(), pos_mask.cuda()

        anchor_mask = torch.unsqueeze(anchor_mask,dim=1)
        pos_mask = torch.unsqueeze(pos_mask, dim=1)
        logits = model(anchor_img, pos_img, neg_img, pos_mask)
        loss_val, cluster_loss, loss_bce = model.get_loss(logits, anchor_mask)

        loss_val_float = loss_val.data.item()

        losses.update(loss_val_float, 1)

        out_str = '%d, %.4f\n'%(count, loss_val_float)
        log_file.write(out_str)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


        count += 1
        if count%args.disp_interval == 0:
            print('Step:%d \t Loss:%.3f '%(count, losses.avg))
            # print('Step:%d \t Loss:%.3f \t '
            #       'Part1: %.3f \t Part2: %.3f'%(count, losses.avg,
            #                             cluster_loss.cpu().data.numpy() if isinstance(cluster_loss, torch.Tensor) else cluster_loss,
            #                             loss_bce.cpu().data.numpy() if isinstance(loss_bce, torch.Tensor) else loss_bce))


        if count%args.save_interval == 0 and count >0:
            save_checkpoint(args,
                            {
                                'global_counter': count,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            }, is_best=False if losses.avg>=0.1 else True ,
                            filename='step_%d.pth.tar'%(count))
    log_file.close()

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    train(args)
