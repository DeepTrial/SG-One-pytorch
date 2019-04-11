# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import PIL
import PIL.Image as Image
from Utils.util import *
np.random.seed(1234)

ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')

class imdb(object):
    """Image database."""

    def __init__(self, name, classes=None):
        self._name = name
        self._num_classes = 0
        if not classes:
            self._classes = []
        else:
            self._classes = classes
        self._image_index = []
        # self.group=0
        # self.num_folds= 4
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.default_roidb

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')   # method=self.gt_roidb
        self.roidb_handler = method

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()  # call self.gt_roidb()
        return self._roidb

    # @property
    # def cache_path(self):
    #     cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
    #     if not os.path.exists(cache_path):
    #         os.makedirs(cache_path)
    #     return cache_path

    @property
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def image_id_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def get_pair_images(self):
        self.group=0
        self.num_folds= 4
        cats = self.get_cats(self.split, self.fold)
        rand_cat = np.random.choice(cats, 1, replace=False)[0]
        sample_img_ids = np.random.choice(len(self.grouped_imgs[rand_cat]), 2, replace=False)
        return (self.grouped_imgs[rand_cat][sample_img_ids[0]],
                self.grouped_imgs[rand_cat][sample_img_ids[1]])

    def get_triple_images(self, split, group, num_folds=4):
        cats = self.get_cats(split, group, num_folds)
        rand_cat = np.random.choice(cats, 2, replace=False)
        sample_img_ids_1 = np.random.choice(len(self.grouped_imgs[rand_cat[0]]), 2, replace=False)
        sample_img_ids_2 = np.random.choice(len(self.grouped_imgs[rand_cat[1]]), 1, replace=False)

        anchor_img = self.grouped_imgs[rand_cat[0]][sample_img_ids_1[0]]
        pos_img = self.grouped_imgs[rand_cat[0]][sample_img_ids_1[1]]
        neg_img = self.grouped_imgs[rand_cat[1]][sample_img_ids_2[0]]

        return (anchor_img, pos_img, neg_img)

    def get_multiclass_train(self, split, group, num_folds=4):
        cats = self.get_cats('train', group, num_folds)
        rand_cat = np.random.choice(cats, 1, replace=False)[0]
        cat_list = self.multiclass_grouped_imgs[rand_cat]
        sample_img_ids_1 = np.random.choice(len(cat_list), 2, replace=False)
        query_img = cat_list[sample_img_ids_1[0]]
        support_img = cat_list[sample_img_ids_1[1]]
        return query_img, support_img, rand_cat

    def get_multiclass_val(self, split, group, num_folds=4):
        cats = self.get_cats('val', group, num_folds)
        rand_cat = np.random.choice(cats, 1, replace=False)[0]
        cat_list = self.multiclass_grouped_imgs[rand_cat]
        sample_img_ids_1 = np.random.choice(len(cat_list), 1, replace=False)[0]
        query_img = cat_list[sample_img_ids_1]
        sup_img_list=[]
        for cat_id in cats:
            cat_list = self.grouped_imgs[cat_id]
            sample_img_ids_1 = np.random.choice(len(cat_list), 1, replace=False)[0]
            img_dict = cat_list[sample_img_ids_1]
            sup_img_list.append(img_dict)
        return (query_img, sup_img_list)


    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in range(self.num_images)]


########################################################################### Read DBs into DBItems ################################################################################
class DAVIS:
    def __init__(self, cfg):
        self.cfg = cfg

    # DAVIS: 1376 Test, 2079 Training
    # Jump-Cut: ?
    def getItems(self, sets, categories=None):
        if isinstance(sets, str):
            sets = [sets]
        if isinstance(categories, str):
            categories = [categories]
        if len(sets) == 0:
            return []
        with open(self.cfg['DB_INFO'], 'r') as f:
            db_info = yaml.load(f)
        sequences = [x for x in db_info['sequences'] if
                     x['set'] in sets and (categories is None or x['name'] in categories)]
        assert len(sequences) > 0

        items = []
        for seq in sequences:
            name = seq['name']
            img_root = osp.join(self.cfg['SEQUENCES_DIR'], name)
            ann_root = osp.join(self.cfg['ANNOTATION_DIR'], name)
            item = DBDAVISItem(name, img_root, ann_root, seq['num_frames'])
            items.append(item)
        return items


class COCO:
    def __init__(self, db_path, dataType):
        self.pycocotools = __import__('pycocotools.coco')
        if dataType == 'training':
            dataType = 'train2014'
        elif dataType == 'test':
            dataType = 'val2014'
        else:
            raise Exception('split \'' + dataType + '\' is not valid! Valid splits: training/test')

        self.db_path = db_path
        self.dataType = dataType

    def getItems(self, cats=[], areaRng=[], iscrowd=False):

        annFile = '%s/annotations/instances_%s.json' % (self.db_path, self.dataType)

        coco = self.pycocotools.coco.COCO(annFile)
        catIds = coco.getCatIds(catNms=cats);
        anns = coco.getAnnIds(catIds=catIds, areaRng=areaRng, iscrowd=iscrowd)
        cprint(str(len(anns)) + ' annotations read from coco', bcolors.OKGREEN)

        items = []
        for i in range(len(anns)):
            ann = anns[i]
            item = DBCOCOItem('coco-' + self.dataType + str(i), self.db_path, self.dataType, ann, coco,
                              self.pycocotools)
            items.append(item)
        return items


class PASCAL_READ_MODES:
    # Returns list of DBImageItem each has the image and one object instance in the mask
    INSTANCE = 0
    # Returns list of DBImageItem each has the image and the mask for all semantic labels
    SEMANTIC_ALL = 1
    # Returns list of DBImageSetItem each has set of images and corresponding masks for each semantic label
    SEMANTIC = 2


class PASCAL:
    def __init__(self, db_path, dataType):
        if dataType == 'training':
            dataType = 'train'
        elif dataType == 'test':
            dataType = 'val'
        else:
            raise Exception('split \'' + dataType + '\' is not valid! Valid splits: training/test')

        self.db_path = db_path
        classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
        self.name_id_map = dict(zip(classes, range(1, len(classes) + 1)))
        self.id_name_map = dict(zip(range(1, len(classes) + 1), classes))
        self.dataType = dataType

    def getCatIds(self, catNms=[]):
        return [self.name_id_map[catNm] for catNm in catNms]

    def get_anns_path(self, read_mode):
        return osp.join(self.db_path, self.dataType + '_' + str(read_mode) + '_anns.pkl')

    def get_unique_ids(self, mask, return_counts=False, exclude_ids=[0, 255]):
        ids, sizes = np.unique(mask, return_counts=True)
        ids = list(ids)
        sizes = list(sizes)
        for ex_id in exclude_ids:
            if ex_id in ids:
                id_index = ids.index(ex_id)
                ids.remove(ex_id)
                sizes.remove(sizes[id_index])

        assert (len(ids) == len(sizes))
        if return_counts:
            return ids, sizes
        else:
            return ids

    def create_anns(self, read_mode):
        with open(osp.join(self.db_path, 'ImageSets', 'Segmentation', self.dataType + '.txt'), 'r') as f:
            lines = f.readlines()
            names = []
            for line in lines:
                if line.endswith('\n'):
                    line = line[:-1]
                if len(line) > 0:
                    names.append(line)
        anns = []
        for item in names:
            mclass_path = osp.join(self.db_path, 'SegmentationClass', item + '.png')
            mobj_path = osp.join(self.db_path, 'SegmentationObject', item + '.png')
            mclass_uint = np.array(Image.open(mclass_path))
            mobj_uint = np.array(Image.open(mobj_path))
            class_ids = self.get_unique_ids(mclass_uint)
            obj_ids, obj_sizes = self.get_unique_ids(mobj_uint, return_counts=True)

            if read_mode == PASCAL_READ_MODES.INSTANCE:
                for obj_idx in range(len(obj_ids)):
                    class_id = int(np.median(mclass_uint[mobj_uint == obj_ids[obj_idx]]))
                    assert (class_id != 0 and class_id != 255 and obj_ids[obj_idx] != 0 and obj_ids[obj_idx] != 255)
                    anns.append(
                        dict(image_name=item, mask_name=item, object_ids=[obj_ids[obj_idx]], class_ids=[class_id],
                             object_sizes=[obj_sizes[obj_idx]]))
            elif read_mode == PASCAL_READ_MODES.SEMANTIC:
                for class_id in class_ids:
                    assert (class_id != 0 or class_id != 255)
                    anns.append(dict(image_name=item, mask_name=item, class_ids=[class_id]))
            elif read_mode == PASCAL_READ_MODES.SEMANTIC_ALL:
                anns.append(dict(image_name=item, mask_name=item, class_ids=class_ids))
        with open(self.get_anns_path(read_mode), 'wb') as f:
            pickle.dump(anns, f)

    def load_anns(self, read_mode):
        path = self.get_anns_path(read_mode)
        if not osp.exists(path):
            self.create_anns(read_mode)
        with open(path, 'rb') as f:
            anns = pickle.load(f)
        return anns

    def get_anns(self, catIds=[], areaRng=[], read_mode=PASCAL_READ_MODES.INSTANCE):
        if areaRng == []:
            areaRng = [0, np.inf]
        anns = self.load_anns(read_mode)
        if catIds == [] and areaRng == [0, np.inf]:
            return anns

        if read_mode == PASCAL_READ_MODES.INSTANCE:
            filtered_anns = [ann for ann in anns if
                             ann['class_ids'][0] in catIds and areaRng[0] < ann['object_sizes'][0] and
                             ann['object_sizes'][0] < areaRng[1]]
        else:
            filtered_anns = []
            catIds_set = set(catIds)
            for ann in anns:
                class_inter = set(ann['class_ids']) & catIds_set
                # remove class_ids that we did not asked for (i.e. are not catIds_set)
                if len(class_inter) > 0:
                    ann = ann.copy()
                    ann['class_ids'] = sorted(list(class_inter))
                    filtered_anns.append(ann)
        return filtered_anns

    def getItems(self, cats=[], areaRng=[], read_mode=PASCAL_READ_MODES.INSTANCE):
        if len(cats) == 0:
            catIds = self.id_name_map.keys()
        else:
            catIds = self.getCatIds(catNms=cats)
        catIds = np.sort(catIds)

        anns = self.get_anns(catIds=catIds, areaRng=areaRng, read_mode=read_mode)
        cprint(str(len(anns)) + ' annotations read from pascal', bcolors.OKGREEN)

        # rand_ids = np.arange(len(anns))
        # np.random.shuffle(rand_ids)
        # anns = anns[rand_ids.tolist()]

        # random.shuffle(anns)
        items = []

        ids_map = None
        if read_mode == PASCAL_READ_MODES.SEMANTIC_ALL:
            old_ids = catIds
            new_ids = range(1, len(catIds) + 1)
            ids_map = dict(zip(old_ids, new_ids))
        for i in range(len(anns)):
            ann = anns[i]
            img_path = osp.join(self.db_path, 'JPEGImages', ann['image_name'] + '.jpg')
            if read_mode == PASCAL_READ_MODES.INSTANCE:
                mask_path = osp.join(self.db_path, 'SegmentationObject', ann['mask_name'] + '.png')
                item = DBPascalItem('pascal-' + self.dataType + '_' + ann['image_name'] + '_' + str(i), img_path,
                                    mask_path, ann['object_ids'])
            else:
                mask_path = osp.join(self.db_path, 'SegmentationClass', ann['mask_name'] + '.png')
                item = DBPascalItem('pascal-' + self.dataType + '_' + ann['image_name'] + '_' + str(i), img_path,
                                    mask_path, ann['class_ids'], ids_map)
            items.append(item)
        return items

    @staticmethod
    def cluster_items(items):
        clusters = {}
        for i, item in enumerate(items):
            assert (isinstance(item, DBPascalItem))
            item_id = item.obj_ids
            assert (len(item_id) == 1), 'For proper clustering, items should only have one id'
            item_id = item_id[0]
            if item_id in clusters:
                clusters[item_id].append(item)
            else:
                clusters[item_id] = DBImageSetItem('set class id = ' + str(item_id), [item])
        return clusters


########################################################################### DB Items ###################################################################################
class DBVideoItem:
    def __init__(self, name, length):
        self.name = name
        self.length = length

    def read_img(self, img_id):
        pass

    def read_mask(self, img_id):
        pass


class DBDAVISItem(DBVideoItem):
    def __init__(self, name, img_root, ann_root, length):
        DBVideoItem.__init__(self, name, length)
        self.img_root = img_root
        self.ann_root = ann_root

    def read_img(self, img_id):
        file_name = osp.join(self.img_root, '%05d.jpg' % (img_id))
        return read_img(file_name)

    def read_mask(self, img_id):
        file_name = osp.join(self.ann_root, '%05d.png' % (img_id))
        mask = read_mask(file_name)
        return mask

    def read_iflow(self, img_id, step, method):
        if method == 'LDOF':
            if step == 1:
                flow_name = osp.join(self.ann_root, '%05d_inv_LDOF.flo' % (img_id))
            elif step == -1:
                flow_name = osp.join(self.ann_root, '%05d_LDOF.flo' % (img_id))
            else:
                raise Exception('unsupported flow step for LDOF')
        elif method == 'EPIC':
            if step == 1:
                flow_name = osp.join(self.ann_root, '%05d_inv.flo' % (img_id))
            elif step == -1:
                flow_name = osp.join(self.ann_root, '%05d.flo' % (img_id))
            else:
                raise Exception('unsupported flow step for EPIC')
        else:
            raise Exception('unsupported flow algorithm')
        try:
            return read_flo_file(flow_name)
        except IOError as e:
            print("Unable to open file", str(e))  # Does not exist OR no read permissions


class DBImageSetItem(DBVideoItem):
    def __init__(self, name, image_items=[]):
        DBVideoItem.__init__(self, name, len(image_items))
        self.image_items = image_items

    def append(self, image_item):
        self.image_items.append(image_item)
        self.length += 1

    def read_img(self, img_id):
        return self.image_items[img_id].read_img()

    def read_mask(self, img_id):
        return self.image_items[img_id].read_mask()


#####
class DBImageItem:
    def __init__(self, name):
        self.name = name

    def read_mask(self):
        pass

    def read_img(self):
        pass


class DBCOCOItem(DBImageItem):
    def __init__(self, name, db_path, dataType, ann_info, coco_db, pycocotools):
        DBImageItem.__init__(self, name)
        self.ann_info = ann_info
        self.db_path = db_path
        self.dataType = dataType
        self.coco_db = coco_db
        self.pycocotools = pycocotools

    def read_mask(self):
        ann = self.coco_db.loadAnns(self.ann_info)[0]
        img_cur = self.coco_db.loadImgs(ann['image_id'])[0]

        rle = self.pycocotools.mask.frPyObjects(ann['segmentation'], img_cur['height'], img_cur['width'])
        m_uint = self.pycocotools.mask.decode(rle)
        m = np.array(m_uint[:, :, 0], dtype=np.float32)
        return m

    def read_img(self):
        ann = self.coco_db.loadAnns(self.ann_info)[0]
        img_cur = self.coco_db.loadImgs(ann['image_id'])[0]
        img_path = '%s/images/%s/%s' % (self.db_path, self.dataType, img_cur['file_name'])
        return read_img(img_path)


class DBPascalItem(DBImageItem):
    def __init__(self, name, img_path, mask_path, obj_ids, ids_map=None):
        DBImageItem.__init__(self, name)
        self.img_path = img_path
        self.mask_path = mask_path
        self.obj_ids = obj_ids
        if ids_map is None:
            self.ids_map = dict(zip(obj_ids, np.ones(len(obj_ids))))
        else:
            self.ids_map = ids_map

    def read_mask(self, orig_mask=False):
        mobj_uint = np.array(Image.open(self.mask_path))

        if orig_mask:
            return mobj_uint.astype(np.float32)
        m = np.zeros(mobj_uint.shape, dtype=np.float32)
        for obj_id in self.obj_ids:
            m[mobj_uint == obj_id] = self.ids_map[obj_id]
        # m[mobj_uint == 255] = 255
        return m

    def read_img(self):
        # return read_img(self.img_path)
        return read_img(self.img_path), self.img_path

