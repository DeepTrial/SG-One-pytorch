
import cv2
import numpy as np
import os
import torch
from torch.autograd import Variable
import json
import argparse
from tqdm import tqdm
from datasets.datalayer import SSDatalayer
from Utils.Restore import restore_test
from Utils.SaveAtten import SAVE_ATTEN
from Utils.SegScorer import SegScorer,measure_confusion_matrix
from Utils import Metrics
from OneShotModel import *

ROOT_DIR = '/'.join(os.getcwd().split('/'))

save_atten = SAVE_ATTEN()

SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots')
DISP_INTERVAL = 20


def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='sg_one')
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--test_frame",type=str,default="all")
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--restore_step', type=int, default=120000)

    return parser.parse_args()





def get_model(args):
    model = eval(args.arch).OneModel(args)

    model = model.cuda()

    return model


def process_frame(model,num_classes,group,iou_list,tp_list,fn_list,fp_list,hist,scorer):
    datalayer = SSDatalayer(group)
    restore_test(args, model, group)

    for count in tqdm(range(1000)):
        dat = datalayer.dequeue()
        ref_img = dat['second_img'][0]
        query_img = dat['first_img'][0]
        query_label = dat['second_label'][0]
        ref_label = dat['first_label'][0]
        #print(query_img.shape,ref_img.shape, ref_label.shape,ref_label.shape)
        deploy_info = dat['deploy_info']
        #print(deploy_info['first_semantic_labels'][0][0] - 1)
        semantic_label = deploy_info['first_semantic_labels'][0][0] - 1

        ref_img, ref_label = torch.Tensor(ref_img).cuda(), torch.Tensor(ref_label).cuda()
        query_img, query_label = torch.Tensor(query_img).cuda(), torch.Tensor(query_label[0,:,:]).cuda()
        #print(query_img.shape,ref_img.shape, ref_label.shape,ref_label.shape)
        # ref_img = ref_img*ref_label
        ref_img_var, query_img_var = Variable(ref_img), Variable(query_img)
        query_label_var, ref_label_var = Variable(query_label), Variable(ref_label)

        ref_img_var = torch.unsqueeze(ref_img_var,dim=0)
        ref_label_var = torch.unsqueeze(ref_label_var, dim=1)
        query_img_var = torch.unsqueeze(query_img_var, dim=0)
        query_label_var = torch.unsqueeze(query_label_var, dim=0)
        #print(query_img_var.shape,ref_img_var.shape, ref_label_var.shape,ref_label_var.shape)
        logits  = model(query_img_var, ref_img_var, ref_label_var,ref_label_var)

            # w, h = query_label.size()
            # outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
            # out_side = F.softmax(outB_side, dim=1).squeeze()
            # values, pred = torch.max(out_side, dim=0)
        values, pred = model.get_pred(logits, query_img_var)
            
        query_img=query_img.cpu().numpy()
        query_img=np.transpose(query_img,(1,2,0))
        query_img[:,:,0]=query_img[:,:,0]*0.229*255+0.485*255
        query_img[:,:,1]=query_img[:,:,1]*0.224*255+0.456*255
        query_img[:,:,2]=query_img[:,:,2]*0.225*255+0.406*255
        query_img = query_img.astype(np.int32)
        #query_img=cv2.cvtColor(query_img,cv2.COLOR_BGR2RGB)
        cv2.imwrite('./snapshots/onemodel_sg_one/predict_result/'+str(count)+'.jpg',(pred.data.cpu().numpy()*255.0).astype(np.int32))
        cv2.imwrite('./snapshots/onemodel_sg_one/predict_result/'+str(count)+'_org.jpg', query_img)
        cv2.imwrite('./snapshots/onemodel_sg_one/predict_result/'+str(count)+'_label.jpg',(query_label.cpu().numpy()*255.0).astype(np.int32))
        pred = pred.data.cpu().numpy().astype(np.int32)
            
        query_label = query_label.cpu().numpy().astype(np.int32)
        class_ind = int(deploy_info['first_semantic_labels'][0][0])-1 # because class indices from 1 in data layer
        scorer.update(pred, query_label, class_ind+1)
        tp, tn, fp, fn = measure_confusion_matrix(query_label, pred)
        # iou_img = tp/float(max(tn+fp+fn,1))
        tp_list[class_ind] += tp
        fp_list[class_ind] += fp
        fn_list[class_ind] += fn
        # max in case both pred and label are zero
        iou_list = [tp_list[ic] /
                    float(max(tp_list[ic] + fp_list[ic] + fn_list[ic],1))
                    for ic in range(num_classes)]


        tmp_pred = pred
        tmp_pred[tmp_pred>0.5] = class_ind+1
        tmp_gt_label = query_label
        tmp_gt_label[tmp_gt_label>0.5] = class_ind+1
        #print(tmp_pred.shape,query_label.shape)
        hist += Metrics.fast_hist(tmp_pred, query_label, 21)


    print("-------------GROUP %d-------------"%(group))
    print(iou_list)
    class_indexes = range(group*5, (group+1)*5)
    print('Mean:', np.mean(np.take(iou_list, class_indexes)))
    
    return iou_list,tp_list,fn_list,fp_list,hist,scorer
        
        

def val_onegroup(args):
    model = get_model(args)
    model.eval()

    num_classes = 20
    tp_list = [0]*num_classes
    fp_list = [0]*num_classes
    fn_list = [0]*num_classes
    iou_list = [0]*num_classes

    hist = np.zeros((21, 21))

    scorer = SegScorer(num_classes=21)
    iou_list,tp_list,\
        fn_list,fp_list,hist,scorer=process_frame(model,num_classes,\
                                                    args.group,iou_list,tp_list,fn_list,fp_list,hist,scorer)

def val_allframe(args):

    model = get_model(args)
    model.eval()

    num_classes = 20
    tp_list = [0]*num_classes
    fp_list = [0]*num_classes
    fn_list = [0]*num_classes
    iou_list = [0]*num_classes

    hist = np.zeros((21, 21))

    scorer = SegScorer(num_classes=21)

    for group in range(4):
        iou_list,tp_list,\
        fn_list,fp_list,hist,scorer=process_frame(model,num_classes,\
                                                    group,iou_list,tp_list,fn_list,fp_list,hist,scorer)

    print('BMVC IOU', np.mean(np.take(iou_list, range(0,20))))

    miou = Metrics.get_voc_iou(hist)
    print('IOU:', miou, np.mean(miou))


    binary_hist = np.array((hist[0, 0], hist[0, 1:].sum(),hist[1:, 0].sum(), hist[1:, 1:].sum())).reshape((2, 2))
    bin_iu = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
    print('Bin_iu:', bin_iu)

    scores = scorer.score()
    for k in scores.keys():
        print(k, np.mean(scores[k]), scores[k])

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    if args.test_frame=="all":
        val_allframe(args)
    else:
        val_onegroup(args)
