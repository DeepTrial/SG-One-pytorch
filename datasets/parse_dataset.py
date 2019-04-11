from __future__ import print_function
from __future__ import absolute_import

from torch.utils.data import Dataset
import numpy as np
import cv2,operator
from datasets.pascal_voc_seg import pascal_voc_seg
import matplotlib.pyplot as plt

# transfer to the specifical parse function according to the name
# Set up voc_<year>_<split>
__sets = {}
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        # transfer to the function in file pascal_voc_seg
        __sets[name] = (lambda split=split, year=year: pascal_voc_seg(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    print("Grab the dataset: %s"%(name))
    return __sets[name]()



class mydataset(Dataset):

    """implemnet for pytorch data.Dataset"""

    def __init__(self, args, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print("group,num_folds",args.group,args.num_folds)
        if is_train:
            self.img_db = get_imdb(args.dataset_name+'_train')
        self.img_db.get_seg_items(args.group, args.num_folds)
        self.transform = transform
        
        self.split = args.split
        self.count=0
        self.group = args.group
        self.num_folds = args.num_folds
        self.is_train = is_train
        
        self.color_map= \
        [[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],              [64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],[128,64,0],[0,192,0],          [128,192,0],[0,64,128]
        ]
        

    def __len__(self):
        # return len(self.image_list)
        return 100000000
    
    def __getitem__(self, idx):
        if self.split == 'train':
            dat_dicts = self.img_db.get_triple_images(split='train', group=self.group, num_folds=4)
            return self.get_item_single_train(dat_dicts)
        elif self.split == 'random_val':
            dat_dicts = self.img_db.get_triple_images(split='val', group=self.group, num_folds=4)
            return self.get_item_rand_val(dat_dicts)
        elif self.split == 'mlclass_val':
            query_img, sup_img_list = self.img_db.get_multiclass_val(split='val', group=self.group, num_folds=4)
            return self.get_item_mlclass_val(query_img, sup_img_list)
        elif self.split == 'mlclass_train':
            query_img, support_img, category = self.img_db.get_multiclass_train(split='train', group=self.group, num_folds=4)
            return self.get_item_mlclass_train(query_img, support_img, category)


    def read_img(self, path,is_gray=False):
        if is_gray:
            return cv2.imread(path,0)
        else:
            return cv2.imread(path)
            

    def _read_data(self, item_dict):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']
        labels_vanish=item_dict['labels']
        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat=mask_dat[:,:,(2,1,0)]
        height,width=mask_dat.shape[:2]
        for label in labels_vanish:
            for hi in range(height):
                for wj in range(width):
                    temp=(mask_dat[hi,wj]).astype(np.uint8).tolist()
                    if not operator.eq(temp,self.color_map[label]):
                        mask_dat[hi, wj]=[0,0,0]
                    else:
                        mask_dat[hi, wj] = [1, 1, 1]
                        
        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def _read_mlclass_val(self, item_dict):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)

        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def _read_mlclass_train(self, item_dict, category):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat[mask_dat!=category+1] = 0
        mask_dat[mask_dat==category+1] = 1

        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def get_item_mlclass_val(self, query_img, sup_img_list):
        que_img, que_mask = self._read_mlclass_val(query_img)
        supp_img = []
        supp_mask = []
        for img_dit in sup_img_list:
            tmp_img, tmp_mask = self._read_mlclass_val(img_dit)
            supp_img.append(tmp_img)
            supp_mask.append(tmp_mask)

        supp_img_processed = []
        if self.transform is not None:
            que_img = self.transform(que_img)
            for img in supp_img:
                supp_img_processed.append(self.transform(img))

        return que_img, que_mask, supp_img_processed, supp_mask

    def get_item_mlclass_train(self, query_img, support_img, category):
        que_img, que_mask = self._read_mlclass_train(query_img, category)
        supp_img, supp_mask = self._read_mlclass_train(support_img, category)
        if self.transform is not None:
            que_img = self.transform(que_img)
            supp_img = self.transform(supp_img)

        return que_img, que_mask, supp_img, supp_mask

    def get_item_single_train(self,dat_dicts):
        first_img, first_mask = self._read_data(dat_dicts[0])
        second_img, second_mask = self._read_data(dat_dicts[1])
        thrid_img, thrid_mask = self._read_data(dat_dicts[2])
        #cv2.imwrite('./snapshots/third_mask'+str(self.count)+'.jpg',thrid_mask*255.0)
        #cv2.imwrite('./snapshots/third_img'+str(self.count)+'.jpg',thrid_img)
        #self.count=self.count+1
        if self.transform is not None:
            first_img = self.transform(first_img)
            second_img = self.transform(second_img)
            thrid_img = self.transform(thrid_img)

        return first_img, first_mask, second_img,second_mask, thrid_img, thrid_mask

    def get_item_rand_val(self,dat_dicts):
        first_img, first_mask = self._read_data(dat_dicts[0])
        second_img, second_mask = self._read_data(dat_dicts[1])
        thrid_img, thrid_mask = self._read_data(dat_dicts[2])

        if self.transform is not None:
            first_img = self.transform(first_img)
            second_img = self.transform(second_img)
            thrid_img = self.transform(thrid_img)

        # return first_img, first_mask, second_img,second_mask
        return first_img, first_mask, second_img,second_mask, thrid_img, thrid_mask, dat_dicts



