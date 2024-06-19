import os
import scipy.io as scio
import glob
import numpy as np
import cv2
import tqdm

# epi, inf, con, dead
pannuke_col = np.array([[32, 32, 32],
                        [0, 0, 255],
                        [255, 0, 255],
                        [255, 0, 0],
                        [0, 255, 255]])

# mis, inf, epi, con
consep_col = np.array([[32, 32, 32],
                       [0, 255, 255],
                       [255, 0, 255],
                       [0, 0, 255],
                       [255, 0, 0]])

# neut, epi, lym, pla, eos, con
lizard_col = np.array([[32, 32, 32],
                       [0, 128, 255],
                       [0, 0, 255],
                       [255, 0, 255],
                       [255, 255, 0],
                       [128, 0, 0],
                       [255, 0, 0]])

# epi, lym, mac, nuet
monusac_col = np.array([[32, 32, 32],
                        [0, 0, 255],
                        [255, 0, 255],
                        [0, 255, 0],
                        [0, 128, 255]])

col_dict = {'pannuke': pannuke_col,
            'consep': consep_col,
            'monusac': monusac_col,
            'monusac_20x': monusac_col,
            'lizard': lizard_col}

def draw_overlay(ori_img, inst_map, cls_map, col):
    img = np.copy(ori_img).astype(np.uint8)
    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    scale = 3
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    inst_map = cv2.resize(inst_map, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    cls_map = cv2.resize(cls_map, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

    insts = np.unique(inst_map).tolist()
    if 0 in insts:
        insts.remove(0)
    for ins in insts:
        inst_mask = np.copy(inst_map)
        inst_mask[inst_mask != ins] = 0
        cls_num = list(set(cls_map[inst_mask == ins].flatten().tolist()))
        assert len(cls_num) == 1
        inst_mask[inst_mask > 0] = 255
        inst_mask = inst_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in range(len(contours)):
            try:
                color = col[int(cls_num[0])].tolist()
            except:
                continue
            img = cv2.drawContours(img, contours, contour, color, 2, 8)

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return img

################ consep ################
# split = 'Test'
# save_path = os.path.join('/home/data1/my/dataset/nuclei_set_vis_ver1/', 'consep')
# os.makedirs(save_path, exist_ok=True)
# img_list = glob.glob(f'/home/data1/my/dataset/consep/{split}/Images/*.png')
# for img_path in tqdm.tqdm(img_list):
#     edge = np.ones((1000, 5, 3)) * 255
#     img_name = os.path.basename(img_path).split('.')[0]
#     label_path = os.path.join(f'/home/data1/my/dataset/consep/{split}/Labels/', img_name + '.mat')
#     img = cv2.imread(img_path)
#     inst_label = scio.loadmat(label_path)['inst_map']
#     cls_label = scio.loadmat(label_path)['type_map']
#     cls_label[(cls_label == 3) | (cls_label == 4)] = 3
#     cls_label[(cls_label == 5) | (cls_label == 6) | (cls_label == 7)] = 4
#     cls_overlay = draw_overlay(img, inst_label, cls_label, col=col_dict['consep'])
#     img_2_save = np.concatenate([
#                     # img, edge,
#                     cls_overlay],
#                     axis=1)
#     cv2.imwrite(f'{save_path}/{img_name}.png', img_2_save)


# # ################ monusac ################
# split = 'Test'
# save_path = os.path.join('/home/data1/my/dataset/nuclei_set_vis_ver1/', 'monusac')
# os.makedirs(save_path, exist_ok=True)
# img_list = glob.glob(f'/home/data1/my/dataset/monusac/{split}/Images/*.png')
# for img_path in tqdm.tqdm(img_list):
#     img_name = os.path.basename(img_path).split('.')[0]
#     label_path = os.path.join(f'/home/data1/my/dataset/monusac/{split}/Labels/', img_name + '.mat')
#     img = cv2.imread(img_path)
#     edge = np.ones((img.shape[0], 5, 3)) * 255
#     inst_label = scio.loadmat(label_path)['inst_map']
#     cls_label = scio.loadmat(label_path)['type_map']
#
#     cls_overlay = draw_overlay(img, inst_label, cls_label, col=col_dict['monusac'])
#     # cls_overlay = cv2.imread(os.path.join(f'/home/data1/my/dataset/monusac/{split}/overlay/', img_name + '.png'))
#     img_2_save = np.concatenate([
#                     # img, edge,
#                     cls_overlay],
#                     axis=1)
#     cv2.imwrite(f'{save_path}/{img_name}.png', img_2_save)

# ################ pannuke ################
# split = 'Fold_1'
# save_path = os.path.join('/home/data1/my/dataset/nuclei_set_vis_ver1/', 'pannuke')
# os.makedirs(save_path, exist_ok=True)
# img_list = glob.glob(f'/home/data1/my/dataset/pannuke/{split}/Images/*.png')
# for img_path in tqdm.tqdm(img_list):
#     # edge = np.ones((256, 5, 3)) * 255
#     img_name = os.path.basename(img_path).split('.')[0]
#     label_path = os.path.join(f'/home/data1/my/dataset/pannuke/{split}/Labels/', img_name + '.mat')
#     img = cv2.imread(img_path)
#     inst_label = scio.loadmat(label_path)['inst_map']
#     cls_label = scio.loadmat(label_path)['cls_mask']
#     cls_label[cls_label == 5] = 1
#     cls_overlay = draw_overlay(img, inst_label, cls_label, col=col_dict['pannuke'])
#     img_2_save = np.concatenate([
#                     # img, edge,
#                     cls_overlay],
#                     axis=1)
#     cv2.imwrite(f'{save_path}/{split}_{img_name}.png', img_2_save)


############### lizard ################
save_path = os.path.join('/home/data1/my/dataset/nuclei_set_vis_ver1/', 'lizard')
os.makedirs(save_path, exist_ok=True)
img_list = glob.glob(f'/home/data1/my/dataset/lizard/Images/*.png')
for img_path in tqdm.tqdm(img_list):
    edge = np.ones((256, 5, 3)) * 255
    img_name = os.path.basename(img_path).split('.')[0]
    label_path = os.path.join(f'/home/data1/my/dataset/lizard/Labels/', img_name + '.mat')
    img = cv2.imread(img_path)
    inst_label = scio.loadmat(label_path)['inst_map']
    cls_label = scio.loadmat(label_path)['cls_map']
    cls_overlay = draw_overlay(img, inst_label, cls_label, col=col_dict['lizard'])
    img_2_save = np.concatenate([
                    # img,
                    # edge,
                    cls_overlay],
                    axis=1)
    cv2.imwrite(f'{save_path}/{img_name}.png', img_2_save)