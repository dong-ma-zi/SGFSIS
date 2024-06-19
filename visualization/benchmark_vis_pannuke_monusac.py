import os

import numpy as np
import matplotlib.pyplot as plt

# config
ext_set = 'PanNuke'
tsk_set = 'MoNuSAC'

save_dir_path = f'./vis_dir/bench_vis_{ext_set}_{tsk_set}'
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)

#################################### vis aji on different baselines ###############
x_shot = np.arange(6)
# full sup
y_full_sup = [0.635, 0.635, 0.635, 0.635, 0.635, 0.635]
# sup_only
y_sup_only = [0.266, 0.347, 0.452, 0.482, 0.528, 0.591]
# baseline 1
y1 = [0.552, 0.572, 0.599, 0.603, 0.601, 0.619]
# baseline 2
y2 = [0.265, 0.373, 0.497, 0.548, 0.567, 0.620]
# ours
y3 = [0.607, 0.601, 0.608, 0.617, 0.609, 0.626]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(np.arange(len(y1)), y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(np.arange(len(y3)), y3, color='blue', marker='^', label='SGFSL', linewidth=2)

plt.xticks(x_shot, [1, 3, 5, 10, 20, 50], fontsize=18)
# plt.title(f'AJI on {ext_set} -> {tsk_set}')
# plt.xlabel('#Shot', fontsize=14)
# plt.ylabel('AJI', fontsize=14)
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
y_full_sup = [0.533, 0.533, 0.533, 0.533, 0.533, 0.533]
# sup_only
y_sup_only = [0.120, 0.213, 0.271, 0.314, 0.395, 0.466]
# baseline 1
y1 = [0.289, 0.324, 0.381, 0.414, 0.445, 0.487]
# baseline 2
y2 = [0.119, 0.230, 0.315, 0.393, 0.412, 0.481]
# ours
y3 = [0.309, 0.389, 0.422, 0.453, 0.480, 0.504]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(np.arange(len(y1)), y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(np.arange(len(y3)), y3, color='blue', marker='^', label='SGFSL', linewidth=2)

plt.xticks(x_shot, [1, 3, 5, 10, 20, 50], fontsize=18)
# plt.title(f'mPQ+ on {ext_set} -> {tsk_set}')
# plt.xlabel('#Shot', fontsize=14)
# plt.legend(loc='lower right', fontsize=12)
# plt.ylabel('mPQ', fontsize=14)
plt.grid()
plt.tight_layout()
plt.ylim(0.0, 0.55)  # Adjust the limits as needed

# Set yticks with both tick positions and labels
ytick_positions = np.arange(0.0, 0.55, 0.05)
ytick_labels = [f'{tick:.1f}' if int(round(tick, 2) * 100) % 10 == 0 else '' for tick in ytick_positions]
plt.yticks(ytick_positions, ytick_labels, fontsize=18)

plt.savefig(os.path.join(save_dir_path, f'mPQ+_on_{ext_set}_{tsk_set}.png'),
            dpi=400)
plt.clf()


############################################## vis F-score ################################
#################################### vis base F-score on different baselines ##############
x_shot = np.arange(6)
# full sup
y_full_sup = [0.766, 0.766, 0.766, 0.766, 0.766, 0.766]
# sup_only
y_sup_only = [0.310, 0.445, 0.549, 0.630, 0.679, 0.720]
# baseline 1
y1 = [0.575, 0.581, 0.663, 0.681, 0.695, 0.713]
# baseline 2
y2 = [0.310, 0.476, 0.608, 0.670, 0.694, 0.737]
# ours
y3 = [0.603, 0.682, 0.726, 0.733, 0.744, 0.757]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(np.arange(len(y1)), y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(np.arange(len(y3)), y3, color='blue', marker='^', label='SGFSL', linewidth=2)

plt.xticks(x_shot, [1, 3, 5, 10, 20, 50], fontsize=18)
# plt.title(f'Base Class F1-Score on {ext_set} -> {tsk_set}')
# plt.xlabel('#Shot', fontsize=14)
# plt.legend(loc='lower right', fontsize=12)
# plt.ylabel('F-base-Score', fontsize=14)
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
y_full_sup = [0.449, 0.449, 0.449, 0.449, 0.449, 0.449]
# sup_only
y_sup_only = [0.023, 0.075, 0.094, 0.151, 0.235, 0.341]
# baseline 1
y1 = [0.093, 0.137, 0.182, 0.232, 0.294, 0.372]
# baseline 2
y2 = [0.022, 0.092, 0.128, 0.227, 0.252, 0.352]
# ours
y3 = [0.059, 0.191, 0.228, 0.295, 0.345, 0.401]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(np.arange(len(y1)), y1, color='orange', marker='o', label='TransFT', linewidth=2)
plt.plot(x_shot, y2, color='green', marker='+', label='SemiSup', linewidth=2)
plt.plot(np.arange(len(y3)), y3, color='blue', marker='^', label='SGFSL', linewidth=2)

plt.xticks(x_shot, [1, 3, 5, 10, 20, 50], fontsize=18)
# plt.title(f'Novel Class F1-Score on {ext_set} -> {tsk_set}')
# plt.xlabel('#Shot', fontsize=20)
# plt.ylabel('F-novel-Score', fontsize=14)
# plt.legend(loc='lower right', fontsize=14)
plt.grid()
plt.tight_layout()
plt.ylim(0.00, 0.55)  # Adjust the limits as needed

# Set yticks with both tick positions and labels
ytick_positions = np.arange(0.00, 0.55, 0.05)
ytick_labels = [f'{tick:.1f}' if int(round(tick, 2) * 100) % 10 == 0 else '' for tick in ytick_positions]
plt.yticks(ytick_positions, ytick_labels, fontsize=18)

plt.savefig(os.path.join(save_dir_path, f'Novel_Class_F1-Score_on_{ext_set}_{tsk_set}.png'),
            dpi=400)