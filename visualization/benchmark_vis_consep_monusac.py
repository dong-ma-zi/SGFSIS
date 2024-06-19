import os

import numpy as np
import matplotlib.pyplot as plt

# config
ext_set = 'CoNSeP'
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
y1 = [0.469, 0.506, 0.532, 0.553, 0.558, 0.590]
# baseline 2
y2 = [0.265, 0.373, 0.497, 0.548, 0.567, 0.620]
# ours
y3 = [0.524, 0.536, 0.550, 0.555, 0.567, 0.603]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(x_shot, y1, color='orange', marker='o', label='TransFT', linewidth=2)
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
y1 = [0.198, 0.273, 0.318, 0.376, 0.410, 0.445]
# baseline 2
y2 = [0.119, 0.230, 0.315, 0.393, 0.412, 0.481]
# ours
y3 = [0.223, 0.293, 0.336, 0.376, 0.421, 0.463]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(x_shot, y1, color='orange', marker='o', label='TransFT', linewidth=2)
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
y1 = [0.439, 0.541, 0.621, 0.647, 0.693, 0.714]
# baseline 2
y2 = [0.310, 0.476, 0.608, 0.670, 0.694, 0.737]
# ours
y3 = [0.479, 0.593, 0.623, 0.672, 0.683, 0.70]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(x_shot, y1, color='orange', marker='o', label='TransFT', linewidth=2)
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
y1 = [0.03, 0.089, 0.114, 0.215, 0.245, 0.312]
# baseline 2
y2 = [0.022, 0.092, 0.128, 0.227, 0.252, 0.352]
# ours
y3 = [0.013, 0.075, 0.141, 0.193, 0.266, 0.334]

plt.plot(x_shot, y_full_sup, color='black',  linestyle='--', label='FullSup', linewidth=2)
plt.plot(x_shot, y_sup_only, color='red', marker='s', label='FullSup-s', linewidth=2)
plt.plot(x_shot, y1, color='orange', marker='o', label='TransFT', linewidth=2)
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