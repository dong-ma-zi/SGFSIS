import os
import numpy as np
import tqdm
import cv2
# "/home/data1/my/dataset/consep/extracted_mirror/valid/256x256_256x256/overlay/"
# "/home/data1/my/dataset/lizard/test_overlay/"

# ext = 'monusac'
# tsk = 'consep'
#
# save_path = f'./quality_vis/{ext}_2_{tsk}/'
# os.makedirs(save_path, exist_ok=True)
#
# edge1 = np.ones(shape=(160, 10, 3)) * 255
# blank = np.ones(shape=(160, 160, 3)) * 255
# img_mun = 5
# edge2 = np.ones(shape=(10, 160 * (img_mun) + 10 * (img_mun - 1), 3)) * 255
#
# gt_overlay_dir = '/home/data1/my/dataset/consep/extracted_mirror/valid/256x256_256x256/vis_2_draw_overlay/'
# img_list = os.listdir(gt_overlay_dir)
# img_list = [img_name.split('.')[0] for img_name in img_list]
#
# img_dir = {'sup_only': f"../FullSup/exp/{tsk}/",
#            'meanTeacher': f"../SemiSup/exp/{tsk}/",
#            'TransFt': f"../TransFT/exp/ext_{ext}_tsk_{tsk}/",
#            'SGFSL': f"../SGFSL_ours/exp/tsk_finetuning/exttrain_{ext}_tskfinetuning_{tsk}/SGFSL/"
#           }
#
#
# for img_name in tqdm.tqdm(img_list):
#     img_shot_dict = {}
#     # label Images
#     img_orig_overlay = cv2.imread(os.path.join(gt_overlay_dir, img_name + '.png'))
#
#     for shot in [1, 5, 10, 50]:
#
#         img_shot_1 = cv2.imread(
#             os.path.join(img_dir['sup_only'], f'labeledShot_{shot}', 'vis_to_draw', img_name + '.png'))
#         img_shot_2 = cv2.imread(
#             os.path.join(img_dir['meanTeacher'], f'labeledShot_{shot}', 'vis_to_draw', img_name + '.png'))
#
#         x = os.path.join(img_dir['TransFt'], f'labeledShot_{shot}', 'vis_to_draw', img_name + '.png')
#
#         img_shot_3 = cv2.imread(
#             os.path.join(img_dir['TransFt'], f'labeledShot_{shot}', 'vis_to_draw', img_name + '.png'))
#         img_shot_4 = cv2.imread(
#             os.path.join(img_dir['SGFSL'], f'shot_{shot}', 'vis_to_draw', img_name + '.png'))
#         img_shot_dict[f"{shot}"] = np.concatenate([
#                                      img_shot_1, edge1,
#                                      img_shot_2, edge1,
#                                      img_shot_3, edge1,
#                                      img_shot_4, edge1,
#                                      img_orig_overlay,
#                                      ], axis=1)
#
#
#     img_concat = np.concatenate([img_shot_dict['1'], edge2,
#                                  img_shot_dict['5'], edge2,
#                                  img_shot_dict['10'], edge2,
#                                  img_shot_dict['50'],
#                                  ], axis=0)
#
#     cv2.imwrite(os.path.join(save_path, img_name + '.png'), img_concat)

ext = 'monusac_20x'
tsk = 'lizard'

save_path = f'./quality_vis/{ext}_2_{tsk}/'
os.makedirs(save_path, exist_ok=True)

edge1 = np.ones(shape=(128, 10, 3)) * 255
blank = np.ones(shape=(128, 128, 3)) * 255
img_mun = 5
edge2 = np.ones(shape=(10, 128 * (img_mun) + 10 * (img_mun - 1), 3)) * 255

gt_overlay_dir = '/home/data1/my/dataset/lizard/vis_2_draw_overlay/'
img_list = os.listdir(gt_overlay_dir)
img_list = [img_name.split('.')[0] for img_name in img_list]

img_dir = {'sup_only': f"../FullSup/exp/{tsk}/",
           'meanTeacher': f"../SemiSup/exp/{tsk}/",
           'TransFt': f"../TransFT/exp/ext_{ext}_tsk_{tsk}/",
           'SGFSL': f"../SGFSL_ours/exp/tsk_finetuning/exttrain_{ext}_tskfinetuning_{tsk}/SGFSL/"
          }


for img_name in tqdm.tqdm(img_list):
    img_shot_dict = {}
    # label Images
    img_orig_overlay = cv2.imread(os.path.join(gt_overlay_dir, img_name + '.png'))

    for shot in [1, 5, 10, 50]:

        img_shot_1 = cv2.imread(
            os.path.join(img_dir['sup_only'], f'labeledShot_{shot}', 'vis_to_draw', img_name + '.png'))
        img_shot_2 = cv2.imread(
            os.path.join(img_dir['meanTeacher'], f'labeledShot_{shot}', 'vis_to_draw', img_name + '.png'))

        # x = os.path.join(img_dir['TransFt'], f'labeledShot_{shot}', 'vis_to_draw', img_name + '.png')

        img_shot_3 = cv2.imread(
            os.path.join(img_dir['TransFt'], f'labeledShot_{shot}', 'vis_to_draw', img_name + '.png'))
        img_shot_4 = cv2.imread(
            os.path.join(img_dir['SGFSL'], f'shot_{shot}', 'vis_to_draw', img_name + '.png'))
        img_shot_dict[f"{shot}"] = np.concatenate([
                                     img_shot_1, edge1,
                                     img_shot_2, edge1,
                                     img_shot_3, edge1,
                                     img_shot_4[16: -16, 16: -16, :], edge1,
                                     img_orig_overlay,
                                     ], axis=1)


    img_concat = np.concatenate([img_shot_dict['1'], edge2,
                                 img_shot_dict['5'], edge2,
                                 img_shot_dict['10'], edge2,
                                 img_shot_dict['50'],
                                 ], axis=0)

    cv2.imwrite(os.path.join(save_path, img_name + '.png'), img_concat)