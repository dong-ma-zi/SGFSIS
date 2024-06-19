import os.path
import cv2
import numpy as np

root_path = './vis_dir/'
img_dict = {}

tranferring_list = ['MoNuSAC_CoNSeP', 'MoNuSAC_Lizard',
                    'CoNSeP_MoNuSAC', 'PanNuke_MoNuSAC']

for setting in tranferring_list:
    setting_dir = os.path.join(root_path, 'bench_vis_' + setting)

    # mpq
    if os.path.exists(os.path.join(setting_dir, 'mPQ+_on_' +  setting + '.png')):
        pq_vis = cv2.imread(os.path.join(setting_dir, 'mPQ+_on_' +  setting + '.png'))
    else:
        pq_vis = np.ones(shape=(1920, 2560, 3)) * 255
        pq_vis = pq_vis.astype(np.uint8)
    vis_img = pq_vis

    # aji
    if os.path.exists(os.path.join(setting_dir, 'AJI_on_' +  setting + '.png')):
        aji_vis = cv2.imread(os.path.join(setting_dir, 'AJI_on_' +  setting + '.png'))
    else:
        aji_vis = np.ones(shape=(1920, 2560, 3)) * 255
        aji_vis = pq_vis.astype(np.uint8)
    vis_img = np.concatenate([vis_img, aji_vis], axis=0)

    # f-base
    if os.path.exists(os.path.join(setting_dir, 'Base_Class_F1-Score_on_' + setting + '.png')):
        fbase_vis = cv2.imread(os.path.join(setting_dir, 'Base_Class_F1-Score_on_' + setting + '.png'))
    else:
        fbase_vis = np.ones(shape=(1920, 2560, 3)) * 255
        fbase_vis = pq_vis.astype(np.uint8)
    vis_img = np.concatenate([vis_img,fbase_vis], axis=0)

    # f-novel
    if os.path.exists(os.path.join(setting_dir, 'Novel_Class_F1-Score_on_' + setting + '.png')):
        fnovel_vis = cv2.imread(os.path.join(setting_dir, 'Novel_Class_F1-Score_on_' + setting + '.png'))
    else:
        fnovel_vis = np.ones(shape=(1920, 2560, 3)) * 255
        fnovel_vis = pq_vis.astype(np.uint8)
    vis_img = np.concatenate([vis_img, fnovel_vis], axis=0)

    img_dict[setting] = vis_img

img_overall = np.concatenate([img_dict['MoNuSAC_CoNSeP'],
                              img_dict['MoNuSAC_Lizard'],
                              img_dict['CoNSeP_MoNuSAC'],
                              img_dict['PanNuke_MoNuSAC']], axis=1)
cv2.imwrite('overall_bench.png', img_overall)

