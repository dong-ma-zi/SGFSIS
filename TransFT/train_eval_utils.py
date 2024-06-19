import os
import sys
import torch
import numpy as np
from models.QuadBranchNet.post_proc import process
import torch.nn.functional as F
from collections import OrderedDict

from utils.loss import mse_loss, xentropy_loss, dice_loss
from utils.viz_utils import draw_overlay_scaling, draw_overlay_by_index
import cv2
from utils.metrics import run_nuclei_type_stat
from utils.metrics import generate_cls_info
from utils.metrics import get_fast_aji, getmPQ

pannuke_col = np.array([[32, 32, 32],
                        [0, 0, 255],
                        [0, 255, 0],
                        [255, 0, 0],
                        [0, 255, 255],
                        [0, 165, 255]])

consep_col = np.array([[32, 32, 32],
                       [0, 255, 255],
                       [255, 0, 255],
                       [0, 0, 255],
                       [255, 0, 0]])

lizard_col = np.array([[32, 32, 32],
                       [0, 165, 255],
                       [0, 255, 0],
                       [0, 0, 255],
                       [255, 255, 0],
                       [255, 0, 0],
                       [0, 255, 255]])

monusac_col = np.array([[32, 32, 32],
                        [0, 0, 255],
                        [0, 255, 255],
                        [0, 255, 0],
                        [255, 0, 0]])

col_dict = {'pannuke': pannuke_col,
            'consep': consep_col,
            'monusac': monusac_col,
            'monusac_20x': monusac_col,
            'lizard': lizard_col}

loss_opts = {"np": {"bce": 1, "dice": 1},
             "conp": {"con_bce": 1, "con_dice": 1},
             "cenp": {"cen_mse": 10},
             "tp": {"type_bce": 1, "type_dice": 1}}

loss_func_dict = {"bce": xentropy_loss,
                  "dice": dice_loss,
                  "con_bce": xentropy_loss,
                  "con_dice": dice_loss,
                  "cen_mse": mse_loss,
                  "type_bce": xentropy_loss,
                  'type_dice': dice_loss}

def train_one_epoch(model, optimizer, loader_labeled,
                    device, num_type=6):
    model.train()
    accu_loss = {"sup": {"bce": 0,
                         "dice": 0,
                         "con_bce": 0,
                         "con_dice": 0,
                         "cen_mse": 0,
                         "type_bce": 0,
                         'type_dice': 0},
                 "total": 0}

    for data in loader_labeled:

        img = data["img"].to(device).type(torch.float32)

        # modified the format of gt
        true_np = data["np_map"].to(device).type(torch.int64)
        true_np = F.one_hot(true_np, num_classes=2).type(torch.float32)

        true_con = data["con_map"].to(device).type(torch.int64)
        true_con = F.one_hot(true_con, num_classes=2).type(torch.float32)

        # true_cen = data["cen_map"].to(device).type(torch.int64)
        # true_cen = F.one_hot(true_cen, num_classes=2).type(torch.float32)

        true_gau = data["gau_map"].to(device).type(torch.float32)

        true_tp = data["tp_map"].to(device).type(torch.int64)
        true_tp = F.one_hot(true_tp, num_classes=num_type).type(torch.float32)


        true_dict = {"np": true_np,
                     'tp': true_tp,
                     "conp": true_con,
                     'cenp': true_gau}

        pred_dict = model(img)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
        pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)
        pred_dict["conp"] = F.softmax(pred_dict["conp"], dim=-1)
        # pred_dict["cenp"] = F.softmax(pred_dict["conp"], dim=-1)[:, :, :, 1]
        pred_dict["cenp"] = pred_dict["cenp"][:, :, :, 0]

        loss = 0
        # 计算有监督部分的损失函数
        for branch_name in pred_dict.keys():
            for loss_name, loss_weight in loss_opts[branch_name].items():
                loss_func = loss_func_dict[loss_name]
                loss_args = [true_dict[branch_name], pred_dict[branch_name]]
                if loss_name == "msge":
                    loss_args.append(true_dict["np"][..., 1])
                    loss_args.append(device)
                term_loss = loss_func(*loss_args) * loss_weight
                accu_loss["sup"][loss_name] += term_loss.cpu().item()
                loss += term_loss

        accu_loss["total"] += loss.cpu().item()
        if not torch.isfinite(loss.cpu()):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return accu_loss

def evaluate_one_epoch_patch_level(model,
                                   test_loader,
                                   test_dataset,
                                   num_type=5,
                                   device='cuda',
                                   dataset='',
                                   is_draw=True,
                                   num_run=1,
                                   save_dir=''):
    vis_save_dir = os.path.join(save_dir, 'vis_to_draw')
    os.makedirs(vis_save_dir, exist_ok=True)
    model.eval()

    results = []
    for data in test_loader:

        img_l = data["img"].to(device).type(torch.float32)
        true_tp = data["tp_map"].numpy()
        true_inst = data["inst_map"].numpy()


        # cal the prediction results
        pred_dict_ = model(img_l)

        # confirm the branch prediction output order np-hv-tp
        pred_dict = OrderedDict()
        pred_dict['np'] = pred_dict_['np']
        pred_dict['conp'] = pred_dict_['conp']
        pred_dict['cenp'] = pred_dict_['cenp']
        pred_dict['tp'] = pred_dict_['tp']

        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )

        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        pred_dict["conp"] = F.softmax(pred_dict["conp"], dim=-1)[..., 1:]

        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1).detach().cpu().numpy()[0]

        pred_inst, inst_info_dict = process(pred_output, nr_types=num_type)
        results += [(true_inst[0], true_tp[0], pred_output,
                     pred_inst, inst_info_dict)]

    ############################ evalute the performence #################################
    total_aji = 0
    res = []
    true_res = []
    pred_dict = {}
    true_dict = {}
    list_length = len(results)
    for idx, result in enumerate(results):
        ori_img = cv2.imread(test_dataset.img_path_list[idx])
        img_name = os.path.basename(test_dataset.img_path_list[idx]).split('.')[0]
        inst_label, cls_label, pred_output, \
        pred_inst, inst_info_dict = result
        # cls to be ingnored in gt labels (0 means ignore)
        ambiguous_mask = np.ones_like(inst_label).astype(np.bool)

        if dataset == 'monusac' or dataset == 'monusac_20x':
            # order can not be changed
            ambiguous_mask = (cls_label != 5)
            inst_label *= ambiguous_mask
            cls_label *= ambiguous_mask

        # assign cls on the prediction map
        pred_cls = np.zeros_like(pred_inst)
        if inst_info_dict is not None:
            for inst in range(1, int(np.max(pred_inst)) + 1):
                try:
                    pred_cls[pred_inst == inst] = inst_info_dict[inst]['type']
                except:
                    pass

        pred_cls *= ambiguous_mask
        pred_inst *= ambiguous_mask

        pred_dict[img_name] = generate_cls_info(pred_inst, pred_cls)
        true_dict[img_name] = generate_cls_info(inst_label, cls_label)

        res += [np.concatenate([pred_inst[..., None], pred_cls[..., None]], axis=-1)[None, ...]]
        true_res += [np.concatenate([inst_label[..., None], cls_label[..., None]], axis=-1)[None, ...]]

        # draw_overlay

        if is_draw and num_run == 1:

            pred_np = pred_output[:, :, 0]
            pred_np = 255 * np.array(pred_np >= 0.5, dtype=np.int32)
            pred_np = pred_np.astype(np.uint8)
            pred_np = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)

            pred_conp = pred_output[:, :, 1]
            pred_conp = 255 * np.array(pred_conp >= 0.5, dtype=np.int32)
            pred_conp = pred_conp.astype(np.uint8)
            pred_conp = cv2.cvtColor(pred_conp, cv2.COLOR_GRAY2BGR)

            pred_cenp = pred_output[:, :, 2]
            pred_cenp = 255 * np.array(pred_cenp >= 0.5, dtype=np.int32)
            pred_cenp = pred_cenp.astype(np.uint8)
            pred_cenp = cv2.cvtColor(pred_cenp, cv2.COLOR_GRAY2BGR)

            pred_type = pred_output[..., -1]
            pred_type = draw_overlay_by_index(pred_type, col_dict[dataset])

            overlay = draw_overlay_scaling(ori_img, inst_label, cls_label, col_dict[dataset])
            pred_vis = draw_overlay_scaling(ori_img, pred_inst, pred_cls, col_dict[dataset])

            # edge1 = np.ones(shape=(pred_vis.shape[0], 5, 3)) * 255
            # vis = np.concatenate([overlay, edge1,
            #                       pred_vis, edge1,
            #                       pred_np, edge1,
            #                       pred_conp, edge1,
            #                       pred_cenp, edge1,
            #                       pred_type], axis=1)
            # cv2.imwrite(f'{vis_save_dir}/{img_name}.png', vis)

            crop_size = 64
            cv2.imwrite(
                f'{vis_save_dir}/{img_name}.png',
                pred_vis[crop_size: -crop_size, crop_size: -crop_size, :])


        # if empty test patch, ignore the metric
        if np.max(inst_label) == 0 and np.max(pred_inst) == 0:
            aji = 0
            list_length -= 1
        elif np.max(inst_label) != 0 and np.max(pred_inst) == 0:
            aji = 0
            print(f'empty_pred: {img_name}')
        else:
            aji = get_fast_aji(inst_label, pred_inst)
        total_aji += aji

    res = np.concatenate(res)
    true_res = np.concatenate(true_res)

    pq_metrics = getmPQ(res, true_res, nr_classes=num_type-1)
    mpq = pq_metrics['multi_pq+'][0]
    print("mpq: ", mpq)

    # classification metrics
    if dataset == 'consep_20x' or dataset == 'monusac_20x' or dataset == 'lizard':
        _20x = True
    else:
        _20x = False

    results_list = run_nuclei_type_stat(pred_dict, true_dict, _20x=_20x)
    F1_det = results_list[0]
    F1_list = results_list[-int(num_type-1):]
    F1_avg = np.mean(F1_list)

    return total_aji / list_length, F1_det, mpq, F1_list, F1_avg
    # return 0, 0, 0, 0, 0

