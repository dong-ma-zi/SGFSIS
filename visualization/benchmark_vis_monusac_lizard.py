import os

import numpy as np
import matplotlib.pyplot as plt

# config
ext_set = 'MoNuSAC'
tsk_set = 'Lizard'

save_dir_path = f'./vis_dir/bench_vis_{ext_set}_{tsk_set}'
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)

#################################### vis aji on different baselines ###############
x_shot = np.arange(6)
# full sup
y_full_sup = [0.585, 0.585, 0.585, 0.585, 0.585, 0.585]
# sup_only
y_sup_only = [0.294, 0.390, 0.425, 0.476, 0.509, 0.544]
# baseline 1
y1 = [0.425, 0.456, 0.471, 0.485, 0.50, 0.518]
# baseline 2
y2 = [0.298, 0.413, 0.447, 0.502, 0.540, 0.555]
# ours
y3 = [0.546, 0.574, 0.585, 0.599, 0.605, 0.616]

plt.plot(np.arange(len(y_full_sup)), y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(np.arange(len(y1)), y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(np.arange(len(y3)), y3, color='blue', marker='^', label='SGFSL', linewidth=2)

plt.xticks(x_shot, [1, 3, 5, 10, 20, 50], fontsize=18)
# plt.title(f'AJI on {ext_set} -> {tsk_set}')
# plt.xlabel('#Shot', fontsize=14)
# plt.ylabel('AJI', fontsize=18)
# plt.legend(loc='lower right', fontsize=12)
plt.grid()
plt.tight_layout()
plt.ylim(0.20, 0.70)  # Adjust the limits as needed

# Set yticks with both tick positions and labels
ytick_positions = np.arange(0.20, 0.70, 0.05)
ytick_labels = [f'{tick:.1f}' if int(round(tick, 2) * 100) % 10 == 0 else '' for tick in ytick_positions]
plt.yticks(ytick_positions, ytick_labels, fontsize=18)

plt.savefig(os.path.join(save_dir_path, f'AJI_on_{ext_set}_{tsk_set}.png'),
            dpi=400)
plt.clf()


#################################### vis mpq+ on different baselines ##############
x_shot = np.arange(6)
# full sup
y_full_sup = [0.453, 0.453, 0.453, 0.453, 0.453, 0.453]
# sup_only
y_sup_only = [0.122, 0.185, 0.219, 0.273, 0.340, 0.423]
# baseline 1
y1 = [0.172, 0.221, 0.256, 0.277, 0.31, 0.354]
# baseline 2
y2 = [0.124, 0.196, 0.240, 0.294, 0.383, 0.437]
# ours
y3 = [0.256, 0.316, 0.356, 0.391, 0.434, 0.468]

plt.plot(np.arange(len(y_full_sup)), y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(np.arange(len(y1)), y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(np.arange(len(y3)), y3, color='blue', marker='^', label='SGFSL', linewidth=2)

plt.xticks(x_shot, [1, 3, 5, 10, 20, 50], fontsize=18)
# plt.title(f'mPQ+ on {ext_set} -> {tsk_set}')
# plt.xlabel('#Shot', fontsize=14)
# plt.legend(loc='lower right', fontsize=12)
# plt.ylabel('mPQ', fontsize=18)
plt.grid()
plt.tight_layout()
plt.ylim(0.00, 0.55)  # Adjust the limits as needed

# Set yticks with both tick positions and labels
ytick_positions = np.arange(0.00, 0.55, 0.05)
ytick_labels = [f'{tick:.1f}' if int(round(tick, 2) * 100) % 10 == 0 else '' for tick in ytick_positions]
plt.yticks(ytick_positions, ytick_labels, fontsize=18)

plt.savefig(os.path.join(save_dir_path, f'mPQ+_on_{ext_set}_{tsk_set}.png'),
            dpi=400)
plt.clf()


############################################## vis F-score ################################
#################################### vis base F-score on different baselines ##############
x_shot = np.arange(6)
# full sup
y_full_sup = [0.533, 0.533, 0.533, 0.533, 0.533, 0.533]
# sup_only
y_sup_only = [0.218, 0.306, 0.340, 0.386, 0.463, 0.524]
# baseline 1
y1 = [0.282, 0.340, 0.375, 0.407, 0.435, 0.470]
# baseline 2
y2 = [0.220, 0.316, 0.361, 0.419, 0.491, 0.538]
# ours
y3 = [0.367, 0.428, 0.465, 0.498, 0.531, 0.564]

plt.plot(np.arange(len(y_full_sup)), y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(np.arange(len(y1)), y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(np.arange(len(y3)), y3, color='blue', marker='^', label='SGFSL', linewidth=2)

plt.xticks(x_shot, [1, 3, 5, 10, 20, 50], fontsize=18)
# plt.title(f'Base Class F1-Score on {ext_set} -> {tsk_set}')
# plt.xlabel('#Shot', fontsize=14)
# plt.legend(loc='lower right', fontsize=12)
# plt.ylabel('F-base-Score', fontsize=18)
plt.grid()
plt.tight_layout()
plt.ylim(0.20, 0.80)  # Adjust the limits as needed

# Set yticks with both tick positions and labels
ytick_positions = np.arange(0.20, 0.80, 0.05)
ytick_labels = [f'{tick:.1f}' if int(round(tick, 2) * 100) % 10 == 0 else '' for tick in ytick_positions]
plt.yticks(ytick_positions, ytick_labels, fontsize=18)

plt.savefig(os.path.join(save_dir_path, f'Base_Class_F1-Score_on_{ext_set}_{tsk_set}.png'),
            dpi=400)
plt.clf()


#################################### vis novel F-score on different baselines ##############
x_shot = np.arange(6)
# full sup
y_full_sup = [0.469, 0.469, 0.469, 0.469, 0.469, 0.469]
# sup_only
y_sup_only = [0.094, 0.134, 0.161, 0.214, 0.294, 0.419]
# baseline 1
y1 = [0.121, 0.165, 0.219, 0.243, 0.294, 0.351]
# baseline 2
y2 = [0.097, 0.145, 0.189, 0.241, 0.349, 0.430]
# ours
y3 = [0.158, 0.239, 0.298, 0.330, 0.399, 0.443]

plt.plot(np.arange(len(y_full_sup)), y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(np.arange(len(y1)), y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(np.arange(len(y3)), y3, color='blue', marker='^', label='SGFSL', linewidth=2)

plt.xticks(x_shot, [1, 3, 5, 10, 20, 50], fontsize=18)
# plt.title(f'Novel Class F1-Score on {ext_set} -> {tsk_set}')
# plt.xlabel('#Shot', fontsize=20)
# plt.ylabel('F-novel-Score', fontsize=18)
# plt.legend(loc='lower right', fontsize=16)
plt.grid()
plt.tight_layout()
plt.ylim(0.00, 0.55)  # Adjust the limits as needed


# Set yticks with both tick positions and labels
ytick_positions = np.arange(0.00, 0.55, 0.05)
ytick_labels = [f'{tick:.1f}' if int(round(tick, 2) * 100) % 10 == 0 else '' for tick in ytick_positions]
plt.yticks(ytick_positions, ytick_labels, fontsize=18)

plt.savefig(os.path.join(save_dir_path, f'Novel_Class_F1-Score_on_{ext_set}_{tsk_set}.png'),
            dpi=400)