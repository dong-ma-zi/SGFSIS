import os
import cv2
import glob
import random
import pickle
import numpy as np
import tqdm
from PIL import Image
from skimage import color
from scipy import stats
from torch.utils.data import Dataset
import torchvision.transforms as trans
import scipy.io as scio
from .utils import gen_targets, cropping_center
import torch

# consep
typesDict = {1: 't1',
             2: 't2',
             3: 't3',
             4: 't4',
             5: 'empty'}

def color_deconv(img):
    # img: HWC---RGB---PIL or numpy
    img = np.array(img)
    null = np.zeros_like(img[:, :, 0])
    img_hed = color.rgb2hed(img)
    img_h = color.hed2rgb(np.stack((img_hed[:, :, 0], null, null), axis=-1))
    return img_h


def train_split_shot(base_dataset, seed_list, shot=1, dataset='', num_type=5):
    '''
    :param base_dataset:
    :param seed:
    :param ratio:
    split the dataset into set ratio, and save the split file
    '''

    num_data = len(base_dataset)
    all_idx = np.arange(num_data)

    # patch_type_dict = {value: [] for value in typesDict.values()}
    patch_type_dict = {i: [] for i in range(1, num_type + 1)}
    for ind, patch_path in tqdm.tqdm(enumerate(base_dataset.label_path_list)):

        # consep
        patch = scio.loadmat(patch_path)
        cls_map = patch['cls_map']
        inst_map = patch['inst_map']
        cls_map = cropping_center(cls_map, [224, 224])
        inst_map = cropping_center(inst_map, [224, 224])

        clses = cls_map.flatten().tolist()
        clses = [c for c in clses if c != 0]

        if len(clses) > 128: # make sure the patch with sufficient fore-ground
            max_freq = stats.mode(clses, keepdims=True)[0][0]
            inst_map[cls_map != max_freq] = 0
            # filter the patch with corres instance less than 2
            patch_type_idx = max_freq if len(np.unique(inst_map).tolist()) >= 3 else num_type
        else:
            # TODO: chage for any dataset
            patch_type_idx = num_type

        patch_type_dict[patch_type_idx] += [ind]
    # patch_type_dict.pop('empty')
    patch_type_dict.pop(num_type)

    ####################### TODO: refine the sampling dict #####################
    full_list = []
    com_type_list = []
    un_com_type_list = []
    # split the types to 2 gruop: full-dose & not full-dose
    for type in patch_type_dict:
        if len(patch_type_dict[type]) < 100:
            un_com_type_list += [type]
        else:
            com_type_list += [type]
            full_list += patch_type_dict[type]

    idx_2_del = []
    refine_iter = 0
    for unfull_type in tqdm.tqdm(un_com_type_list):
        while len(patch_type_dict[unfull_type]) < 100:
            refine_iter += 1
            for idx in full_list:
                patch = scio.loadmat(base_dataset.label_path_list[idx])
                cls_map = patch['cls_map']
                inst_map = patch['inst_map']
                cls_map = cropping_center(cls_map, [224, 224])
                inst_map = cropping_center(inst_map, [224, 224])

                clses = cls_map.flatten().tolist()
                clses = [c for c in clses if c != 0]

                # get the k-iter max freq categories
                for _ in range(refine_iter):
                    max_freq = stats.mode(clses, keepdims=True)[0][0]
                    clses = [c for c in clses if c != max_freq]

                max_freq = stats.mode(clses, keepdims=True)[0][0]
                inst_map[cls_map != max_freq] = 0
                # filter the patch with corres instance less than 2
                patch_type_idx = max_freq if len(np.unique(inst_map).tolist()) >= 3 else num_type
                if patch_type_idx == unfull_type:
                    patch_type_dict[unfull_type] += [idx]
                    idx_2_del += [idx]

    for type in com_type_list:
        for idx in idx_2_del:
            if idx in patch_type_dict[type]:
                patch_type_dict[type].remove(idx)
    ############################################################################

    # ############################ TODO: draw lizard #############################
    # # lizard
    # colorDict = {1: [0, 165, 255],
    #              2: [0, 255, 0],
    #              3: [0, 0, 255],
    #              4: [255, 255, 0],
    #              5: [255, 0, 0],
    #              6: [0, 255, 255]}
    #
    # typesDict = {1: 'neutrophil',
    #              2: 'epithelial',
    #              3: 'lymphocyte',
    #              4: 'plasma',
    #              5: 'eosinophil',
    #              6: 'connective',
    #              7: 'empty'}
    #
    # for type in patch_type_dict:
    #     os.makedirs(os.path.join(f'../rand_split_file_{dataset}/labeled_shot_{shot}', typesDict[type]))
    #     for idx in patch_type_dict[type]:
    #         img = cv2.imread(base_dataset.img_path_list[idx])
    #         save_name = os.path.basename(base_dataset.img_path_list[idx]).split('.')[0]
    #         patch = scio.loadmat(base_dataset.label_path_list[idx])
    #         cls_map = patch['cls_map']
    #         inst_map = patch['inst_map']
    #         # get the nuclei instance
    #         insts = np.unique(inst_map).tolist()
    #         insts.remove(0)
    #
    #         for ins in insts:
    #             inst_mask = np.copy(inst_map)
    #             inst_mask[inst_mask != ins] = 0
    #
    #             cls_num = list(set(cls_map[inst_mask == ins].flatten().tolist()))
    #             if len(cls_num) != 1:
    #                 print('wrong!!!')
    #
    #             inst_mask[inst_mask > 0] = 255
    #             inst_mask = inst_mask.astype(np.uint8)
    #             contours, hierarchy = cv2.findContours(inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #             for contour in range(len(contours)):
    #                 color = colorDict[cls_num[0]]
    #                 img = cv2.drawContours(img, contours, contour, color, 1, 8)
    #         cv2.imwrite(os.path.join(f'../rand_split_file_{dataset}/labeled_shot_{shot}', typesDict[type],
    #                     save_name + '.png'), img)
    # ############################################################################################


    for i, seed in enumerate(seed_list, start=1):
        random.seed(seed)
        labeled_idx = []
        for type in patch_type_dict:
            labeled_idx += random.sample(patch_type_dict[type], shot)
        # np.random.shuffle(labeled_idx)
        unlabeled_idx = list(set(all_idx) - set(labeled_idx))
        split = {
            'labeled': np.array(labeled_idx),
            'unlabeled': np.array(unlabeled_idx)
        }
        with open(f'../rand_split_file_{dataset}/labeled_shot_{shot}/split_{i}.pkl', 'wb') as f:
            pickle.dump(split, f)


class QuadRandomFlip():
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask1=None, mask2=None, mask3=None, mask4=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask1 is not None:
                mask1 = cv2.flip(mask1, d)
            if mask2 is not None:
                mask2 = cv2.flip(mask2, d)
            if mask3 is not None:
                mask3 = cv2.flip(mask3, d)
            if mask3 is not None:
                mask4 = cv2.flip(mask4, d)
        return img, mask1, mask2, mask3, mask4

class QuadRandomRotate():
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask1=None, mask2=None, mask3=None, mask4=None):
        if random.random() < self.prob:
            angle = random.randint(0, 4)
            img = np.rot90(img, angle)
            if mask1 is not None:
                mask1 = np.rot90(mask1, angle)
            if mask2 is not None:
                mask2 = np.rot90(mask2, angle)
            if mask3 is not None:
                mask3 = np.rot90(mask3, angle)
            if mask4 is not None:
                mask4 = np.rot90(mask4, angle)
        return img, mask1, mask2, mask3, mask4

class QuadCompose():
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, mask1=None, mask2=None, mask3=None, mask4=None):
        for t in self.transforms:
            img, mask1, mask2, mask3, mask4 = t(img, mask1, mask2, mask3, mask4)
        return img, mask1, mask2, mask3, mask4

from scipy.ndimage import gaussian_filter
def get_gaussian_map_from_points(points_map, radius):
    if np.sum(points_map > 0):
        label_detect = gaussian_filter(points_map.astype(np.float64), sigma=radius/3)
        val = np.min(label_detect[points_map > 0])
        label_detect = label_detect / val
        label_detect[label_detect < 0.05] = 0
        label_detect[label_detect > 1] = 1
    else:
        label_detect = np.zeros(points_map.shape)
    return label_detect

class Base_Dataset(Dataset):
    def __init__(self, data_root):
        super(Base_Dataset, self).__init__()
        # data_root为数据集的根目录
        # images 中存放图片， Annotations中存放标签
        self.img_path_list = sorted(glob.glob("{}/Images/*.png".format(data_root)))
        self.label_path_list = sorted(glob.glob("{}/Label_cr_1_ct_1/*.mat".format(data_root)))
        # 检查数据是否对的上
        for img_path, label_path in zip(self.img_path_list, self.label_path_list):
            assert os.path.basename(img_path).split('.')[0] == \
                os.path.basename(label_path).split('.')[0], "Image not matching with label"

    def __len__(self):
        return len(self.img_path_list)


class Dataset_labeled(Base_Dataset):
    def __init__(self, data_root, idx, transforms=None, mag='40x'):
        super(Dataset_labeled, self).__init__(data_root)

        # if no idx list was provided, use all data
        if idx == []:
            idx = range(len(self.img_path_list))

        self.img_path_list = [self.img_path_list[i] for i in idx]
        self.label_path_list = [self.label_path_list[i] for i in idx]
        self.transforms = transforms
        self.totensor = trans.ToTensor()
        self.mag = mag

    def __getitem__(self, index):
        image_path, label_path = self.img_path_list[index], self.label_path_list[index]

        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print('{} does not exist.'.format(image_path))
        image = Image.open(image_path).convert('RGB')
        label_cls = scio.loadmat(label_path)['cls_map']
        label_center = scio.loadmat(label_path)['center_map']
        label_contour = scio.loadmat(label_path)['contour_map']
        label_inst = scio.loadmat(label_path)['inst_map']
        raw_label_inst = np.copy(label_inst)


        if self.transforms is not None:
            image, label_cls, label_contour, label_center, label_inst = self.transforms(image,
                                                                                        label_cls,
                                                                                        label_contour,
                                                                                        label_center,
                                                                                        label_inst)
        else:
            image = np.array(image)

        # expand the contour
        label_contour[label_contour != 0] = 1
        if self.mag == '40x':
            kernel = np.ones((3, 3), np.uint8)
            label_contour = cv2.dilate(label_contour, kernel, iterations=1)

        # expand the centroid
        label_center[label_center != 0] = 1
        # get gaussian map of centroid map
        if self.mag == '40x':
            kernel = np.ones((5, 5), np.uint8)
            label_center_gaussian = get_gaussian_map_from_points(label_center, radius=8)
        elif self.mag == '20x':
            kernel = np.ones((3, 3), np.uint8)
            label_center_gaussian = get_gaussian_map_from_points(label_center, radius=6)
        # kernel = np.ones((3, 3), np.uint8)
        label_center = cv2.dilate(label_center, kernel, iterations=1)
        label_fore = np.array(label_cls > 0, dtype=np.int32)
        image = self.totensor(image)

        # return image, \
        #     torch.Tensor(label_fore), \
        #     torch.Tensor(label_contour), \
        #     torch.Tensor(label_center), \
        #     torch.Tensor(label_center_gaussian), \
        #     torch.Tensor(label_cls), \
        #     torch.tensor(raw_label_inst), \


        return {"img": image,
                "np_map": label_fore,
                "con_map": label_contour,
                "cen_map": label_center,
                "gau_map": label_center_gaussian,
                "tp_map": np.int32(label_cls),
                "inst_map": np.int32(raw_label_inst)}

    # def __getitem__(self, idx):
    #     img = Image.open(self.img_path_list[idx]).convert('RGB')
    #     inst_label = scio.loadmat(self.label_path_list[idx])['inst_map']
    #     raw_inst_map = np.copy(inst_label)
    #     cls_label = scio.loadmat(self.label_path_list[idx])['cls_map']
    #
    #     if self.transforms is not None:
    #         img, inst_label, cls_label = self.transforms(img,
    #                                                      inst_label,
    #                                                      cls_label)
    #     else:
    #         img = np.array(img)
    #
    #     if self.mask_shape is not None:
    #         img = cropping_center(img, self.mask_shape)
    #         cls_label = cropping_center(cls_label, self.mask_shape)
    #         inst_label = gen_targets(np.int32(inst_label), self.mask_shape)
    #     else:
    #         inst_label = gen_targets(np.int32(inst_label), [img.shape[0], img.shape[1]])
    #     img = self.totensor(img)
    #
    #     return {"img": img,
    #             "np_map": inst_label["np_map"],
    #             "hv_map": inst_label["hv_map"],
    #             "tp_map": cls_label,
    #             "inst_map": np.int32(raw_inst_map)}

    def __len__(self):
        return len(self.img_path_list)


class Dataset_unlabeled(Base_Dataset):
    def __init__(self, data_root, idx):
        super(Dataset_unlabeled, self).__init__(data_root)
        self.img_path_list = [self.img_path_list[i] for i in idx]
        self.weakly_DA = trans.Compose([
            trans.RandomHorizontalFlip(0.5),
            trans.RandomVerticalFlip(0.5),
            trans.ToTensor(),
        ])
        self.strongly_DA = trans.Compose([
            trans.ColorJitter(0.3, 0.3, 0.3, 0),
            trans.RandomGrayscale(0.1)
        ])

    def __getitem__(self, idx):
        img = Image.open(self.img_path_list[idx]).convert('RGB')
        # img = cropping_center(np.array(img), self.mask_shape)

        img_u_w = self.weakly_DA(img)
        img_u_s = self.strongly_DA(img_u_w)
        return {"img_u_w": img_u_w,
                "img_u_s": img_u_s}

    def __len__(self):
        return len(self.img_path_list)


