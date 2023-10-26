from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
# import PIL
import numpy as np
import pickle
import operator
from .Imdb import imdb
from collections import OrderedDict
import matplotlib.pyplot as plt
# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete


class PASCAL_READ_MODES:
    #Returns list of ImageItem each has the image and one object instance in the mask
    INSTANCE = 0
    #Returns list of ImageItem each has the image and the mask for all semantic labels
    SEMANTIC_ALL = 1
    #Returns list of ImageSetItem each has set of images and corresponding masks for each semantic label
    SEMANTIC = 2

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

    
class pascal_voc_seg(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._data_path = cfg.Data_Path
        
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)
        
        self._devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        self._classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self.color_map= \
        [[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],              [64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],[128,64,0],[0,192,0],          [128,192,0],[0,64,128]
        ]
        
        self._image_ids = self._load_image_set_ids()
        
        self.read_mode = PASCAL_READ_MODES.SEMANTIC_ALL
        # self.get_seg_items()


    def get_seg_items(self, group, num_folds=4):
        pkl_file = os.path.join(self._data_path, 'cache', 'aaai_pascal_voc_seg_img_db.pkl')
        if os.path.exists(pkl_file):
            try:
                with open(pkl_file,'rb') as f:
                    self.img_db = pickle.load(f)
                    print("read file 'aaai_pascal_voc_seg_img_db.pkl' successful!")
            except:
                print("open file 'aaai_pascal_voc_seg_img_db.pkl' failed!")
        else:
            print("generate file 'aaai_pascal_voc_seg_img_db.pkl'")
            self.img_db = self.getItems()

            if not os.path.exists(os.path.join(self._data_path, 'cache')):
                os.mkdir(os.path.join(self._data_path, 'cache'))
            with open(pkl_file, 'wb') as f:
                pickle.dump(self.img_db, f)

        self.get_seg_items_single_clalss()
        self.get_seg_items_multiclalss(group, num_folds)

        print('Total images: %d'%(len(self.img_db)))



    def get_seg_items_single_clalss(self):
        self.single_img_db = self.filter_single_class_img(self.img_db)
        print('Total images after filtering: %d'%(len(self.single_img_db)))
        self.grouped_imgs = self.group_images(self.single_img_db)

    def get_seg_items_multiclalss(self, group, num_folds):
        train_cats = self.get_cats('train', group, num_folds)
        val_cats = self.get_cats('val', group, num_folds)
        print('Train Categories:', train_cats)
        print('Val Categories:', val_cats)

        multiclass_grouped_imgs = self.group_images(self.img_db)
        # for cat_id, key in enumerate(multiclass_grouped_imgs.keys()):
        #     print(len(multiclass_grouped_imgs[key]))
        self.multiclass_grouped_imgs = self.filter_multi_class_img(multiclass_grouped_imgs, train_cats, val_cats)

        # for cat_id, key in enumerate(self.multiclass_grouped_imgs.keys()):
        #     print('after filter:',len(self.multiclass_grouped_imgs[key]))


    def getItems(self):
        ''' walk over all the segmentation image and mask'''
        items = []
        for i in range(len(self._image_ids)):
            item = {}
            item['img_id'] = self._image_ids[i]      # image name      
            item['img_path'] = self.img_path_at(i)   # image path
            item['mask_path'] = self.mask_path_at(i) # mask path
            item['labels'] = self.get_labels(item['mask_path'])  #to determine what category is in the image
            items.append(item)

        return items
    
def get_labels(self, mask_path):
        mask = self.read_mask(mask_path) 
        height,width=mask.shape[:2]
        labels=[]
        for hi in range(height):                # associate the RGB color of every pixel to get the corresponding label 
            for wj in range(width): 
                temp=(mask[hi,wj]*256).astype(np.uint8).tolist()
                for l in range(20):
                    if operator.eq(temp[:-1],self.color_map[l]):
                        labels.append(l)
        labels=np.unique(labels).tolist()
        return labels

    def mask_path_at(self, i):
        if self.read_mode == PASCAL_READ_MODES.INSTANCE:
            mask_path = os.path.join(self._data_path, 'SegmentationObject', self._image_ids[i]+'.png')
        else:
            mask_path =  os.path.join(self._data_path, 'SegmentationClass', self._image_ids[i]+'.png')

        return mask_path

    def img_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_ids[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i


    def filter_multi_class_img(self, grouped_dict_list, train_cats, val_cats):
        grouped_imgs = OrderedDict()
        for key in grouped_dict_list.keys():
            grouped_imgs[key] = []

        for key in grouped_dict_list.keys():
            cat_list = grouped_dict_list[key]
            for img_dict in cat_list:
                if key in set(train_cats):
                    labels = img_dict['labels']
                    if set(labels).issubset(train_cats):
                        grouped_imgs[key].append(img_dict)
                elif key in set(val_cats):
                    labels = img_dict['labels']
                    if set(labels).issubset(val_cats):
                        grouped_imgs[key].append(img_dict)
        
        return grouped_imgs


    def filter_single_class_img(self, img_db):
        '''count images which have only one class'''
        filtered_db = []
        for img_dict in img_db:
            if len(img_dict['labels']) == 1:
                filtered_db.append(img_dict)
        return filtered_db


    def group_images(self, img_db):
        '''
        Images of the same label cluster to one list
        Images with multicalsses will be copyed to each class's list
        :return:
        '''
        grouped_imgs = OrderedDict()
        for cls in self._classes:
            grouped_imgs[self._class_to_ind[cls]] = []
        for img_dict in img_db:
            for label in img_dict['labels']:
                grouped_imgs[label].append(img_dict)

        return grouped_imgs

    def read_mask(self, mask_path):
        assert os.path.exists(mask_path), "%s does not exist"%(mask_path)
        mask = plt.imread(mask_path)
        return mask

   

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def get_cats(self, split, group,  num_folds):
        '''
          Returns a list of categories (for training/test) for a given fold number

          Inputs:
            split: specify train/val
            fold : fold number, out of num_folds
            num_folds: Split the set of image classes to how many folds. In BMVC paper, we use 4 folds

        '''
        num_cats = self.num_classes
        assert(num_cats%num_folds==0)
        val_size = int(num_cats/num_folds)
        assert(group<num_folds), 'group: %d, num_folds:%d'%(group, num_folds)
        val_set = [group*val_size+v for v in range(val_size)]
        train_set = [x for x in range(num_cats) if x not in val_set]
        if split=='train':
            # print('Classes for training:'+ ','.join([self.classes[x] for x in train_set]))
            return train_set
        elif split=='val':
            # print('Classes for testing:'+ ','.join([self.classes[x] for x in train_set]))
            return val_set


    def split_id(self, path):
        return path.split()[0].split('/')[-1].split('.')[0]

    def _load_image_set_ids(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/train.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets/Segmentation',self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [self.split_id(x) for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

