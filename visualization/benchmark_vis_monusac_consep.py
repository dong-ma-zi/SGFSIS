import os

import numpy as np
import matplotlib.pyplot as plt

# config
ext_set = 'MoNuSAC'
tsk_set = 'CoNSeP'

save_dir_path = f'./vis_dir/bench_vis_{ext_set}_{tsk_set}'
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)

#################################### vis aji on different baselines ###############
x_shot = np.arange(6)
# full sup
y_full_sup = [0.531, 0.531, 0.531, 0.531, 0.531, 0.531]
# sup_only
y_sup_only = [0.252, 0.357, 0.413, 0.439, 0.469, 0.504]
# baseline 1
y1 = [0.441, 0.470, 0.490, 0.501, 0.510, 0.518]
# baseline 2
y2 = [0.249, 0.371, 0.430, 0.461, 0.485, 0.518]
# ours
y3 = [0.462, 0.482, 0.503, 0.510, 0.516, 0.518]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(x_shot, y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(x_shot, y3, color='blue', marker='^', label='SGFSL', linewidth=2)

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
y_full_sup = [0.382, 0.382, 0.382, 0.382, 0.382, 0.382]
# sup_only
y_sup_only = [0.12, 0.189, 0.235, 0.263, 0.310, 0.340]
# baseline 1
y1 = [0.216, 0.268, 0.291, 0.314, 0.338, 0.345]
# baseline 2
y2 = [0.121, 0.208, 0.252, 0.291, 0.316, 0.343]
# ours
y3 = [0.244, 0.277, 0.305, 0.321, 0.343, 0.353]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(x_shot, y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(x_shot, y3, color='blue', marker='^', label='SGFSL', linewidth=2)

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
y_full_sup = [0.591, 0.591, 0.591, 0.591, 0.591, 0.591]
# sup_only
y_sup_only = [0.296, 0.370, 0.431, 0.488, 0.516, 0.552]
# baseline 1
y1 = [0.397, 0.450, 0.491, 0.508, 0.546, 0.54]
# baseline 2
y2 = [0.289, 0.396, 0.454, 0.527, 0.541, 0.564]
# ours
y3 = [0.452, 0.469, 0.504, 0.521, 0.539, 0.559]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(x_shot, y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(x_shot, y3, color='blue', marker='^', label='SGFSL', linewidth=2)

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
y_full_sup = [0.381, 0.381, 0.381, 0.381, 0.381, 0.381]
# sup_only
y_sup_only = [0.119, 0.207, 0.229, 0.271, 0.321, 0.332]
# baseline 1
y1 = [0.158, 0.234, 0.242, 0.306, 0.337, 0.357]
# baseline 2
y2 = [0.119, 0.212, 0.227, 0.294, 0.305, 0.328]
# ours
y3 = [0.167, 0.260, 0.279, 0.299, 0.337, 0.362]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(x_shot, y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(x_shot, y3, color='blue', marker='^', label='SGFSL', linewidth=2)

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