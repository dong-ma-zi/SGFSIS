import glob
import os
import shutil
import pandas as pd
import tqdm

# ------------------ train /test set --------------------- #
# fold = 'Test'
# csv_file = "/home/data1/my/dataset/lizard/test_exc_consep.csv"
# df = pd.read_csv(csv_file)
#
# file_list = list(df['Image'])
# file_list = [path.split('.')[0] for path in file_list]
#
# os.makedirs(f'/home/data1/my/dataset/lizard/{fold}/Label_cr_1_ct_1')
# os.makedirs(f'/home/data1/my/dataset/lizard/{fold}/Images')
#
# for file_name in file_list:
#     shutil.copyfile(f'/home/data1/my/dataset/lizard/Label_cr_1_ct_1/{file_name}.mat',
#                     f'/home/data1/my/dataset/lizard/{fold}/Label_cr_1_ct_1/{file_name}.mat')
#     shutil.copyfile(f'/home/data1/my/dataset/lizard/Images/{file_name}.png',
#                     f'/home/data1/my/dataset/lizard/{fold}/Images/{file_name}.png')

image_save_dir = '/home/data1/my/dataset/pannuke/Fold_12/Images/'
label_save_dir = '/home/data1/my/dataset/pannuke/Fold_12/Label_cr_1_ct_1/'
for fold in [1, 2]:
    image_list = os.listdir(f"/home/data1/my/dataset/pannuke/Fold_{fold}/Images")
    file_list = [path.split('.')[0] for path in image_list]
    for file_name in tqdm.tqdm(file_list):
        shutil.copyfile(os.path.join(f'/home/data1/my/dataset/pannuke/Fold_{fold}/Images/',
                                     file_name + '.png'),
                        os.path.join(image_save_dir, f'Fold_{fold}_' + file_name + '.png'))
        shutil.copyfile(os.path.join(f'/home/data1/my/dataset/pannuke/Fold_{fold}/Label_cr_1_ct_1/',
                                     file_name + '.mat'),
                        os.path.join(label_save_dir, f'Fold_{fold}_' + file_name + '.mat'))