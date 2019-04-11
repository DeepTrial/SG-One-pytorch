from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.train = edict()
__C.train.LR = 1e-5
__C.train.batch_size=1
__C.train.max_steps = 1000000
__C.train.group = 0
__C.train.save_interval=10000
__C.train.disp_interval=200
#
# Testing options
#
__C.test = edict()


# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Download Pascal VOC from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
__C.DataSet_Name='voc_2012'
__C.PASCAL_PATH= '/home/cc/workbook/One-Shot/SG-One/data/VOCdevkit2012/VOC2012/'
__C.COCO_PATH = ''
__C.SBD_PATH = ''
__C.Data_Path=__C.PASCAL_PATH

