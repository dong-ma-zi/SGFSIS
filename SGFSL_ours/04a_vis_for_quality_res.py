# import logging
# import os
# import random
# import cv2
# import numpy as np
# import argparse
# import torch
# import torch.backends.cudnn as cudnn
# import glob
# import torch.nn.parallel
# import torch.optim
# import torch.utils.data
# from scipy import ndimage
# import torch.nn.functional as F
# from collections import OrderedDict
# from models.post_proc import process
# from models.SGFSL import SGFSL
# from utils.loss import xentropy_loss, dice_loss, mse_loss
# from dataset.nuclei_dataset import Tsk_Finetune_Dataset
# from utils import config
# from utils.viz_utils import draw_overlay_scaling, draw_overlay_by_index, draw_rand_inst_overlay_scaling, rand_color_visualization
# from utils.metrics import run_nuclei_type_stat
# from utils.metrics import generate_cls_info
# from utils.metrics import get_fast_aji, getmPQ, get_dice
#
# cv2.ocl.setUseOpenCL(False)
# cv2.setNumThreads(0)
#
# pannuke_col = np.array([[32, 32, 32],
#                         [0, 0, 255],
#                         [0, 255, 0],
#                         [255, 0, 0],
#                         [0, 255, 255],
#                         [0, 165, 255]])
#
# consep_col = np.array([[32, 32, 32],
#                        [0, 255, 255],
#                        [255, 0, 255],
#                        [0, 0, 255],
#                        [255, 0, 0]])
#
# lizard_col = np.array([[32, 32, 32],
#                        [0, 165, 255],
#                        [0, 255, 0],
#                        [0, 0, 255],
#                        [255, 255, 0],
#                        [255, 0, 0],
#                        [0, 255, 255]])
#
# monusac_col = np.array([[32, 32, 32],
#                         [0, 0, 255],
#                         [0, 255, 255],
#                         [0, 255, 0],
#                         [255, 0, 0]])
#
# col_dict = {'pannuke': pannuke_col,
#             'consep': consep_col,
#             'monusac': monusac_col,
#             'lizard': lizard_col}
#
# loss_func_dict = {"bce": xentropy_loss,
#                   "dice": dice_loss,
#                   "con_bce": xentropy_loss,
#                   "con_dice": dice_loss,
#                   "cen_mse": mse_loss,
#                   "type_bce": xentropy_loss,
#                   'type_dice': dice_loss}
#
# loss_opts = {"np": {"bce": 1, "dice": 1},
#              "conp": {"con_bce": 1, "con_dice": 1},
#              "cenp": {"cen_mse": 10},
#              "tp": {"type_bce": 1, "type_dice": 1}}
#
# def get_parser():
#     parser = argparse.ArgumentParser(description='PyTorch Nuclei Instance Segmentation')
#     parser.add_argument('--config', type=str,
#                         # default='./config/tsk_finetuning/ext_consep_tsk_monusac.yaml',
#                         # default='./config/tsk_finetuning/ext_pannuke_tsk_monusac.yaml',
#                         # default='./config/tsk_finetuning/ext_monusac20x_tsk_lizard.yaml',
#                         # default='./config/tsk_finetuning/ext_monusac_tsk_consep.yaml',
#                         default='./config/tsk_finetuning/ext_monusac_tsk_pannuke.yaml',
#                         help='config file')
#     parser.add_argument('opts', help='see .yaml for all options', default=None, nargs=argparse.REMAINDER)
#     args = parser.parse_args()
#     assert args.config is not None
#     cfg = config.load_cfg_from_cfg_file(args.config)
#     if args.opts is not None:
#         cfg = config.merge_cfg_from_list(cfg, args.opts)
#     return cfg
#
# def worker_init_fn(worker_id):
#     random.seed(args.manual_seed + worker_id)
#     np.random.seed(args.manual_seed + worker_id)
#
# def model_init(model_weight):
#
#     model = SGFSL(num_types=args.classes, modal_num=args.modal_num)
#     model.to(device)
#     model = torch.nn.DataParallel(model, device_ids=args.train_gpu)
#
#     ####################################### TODO: load pretrained model #################################
#     model_weight = torch.load(model_weight, map_location='cpu')['model']
#
#     load_weight_dict = {k: v for k, v in model_weight.items()
#                         if model.state_dict()[k].numel() == v.numel()}
#     model.load_state_dict(load_weight_dict, strict=False)
#
#     return model
#
# def main():
#     args = get_parser()
#     assert args.classes > 1
#     torch.cuda.set_device(args.train_gpu[0])
#     global device
#
#     device = torch.device(f'cuda:{args.train_gpu[0]}' if torch.cuda.is_available() else 'cpu')
#     print(device)
#
#     if args.manual_seed is not None:
#         cudnn.benchmark = False
#         cudnn.deterministic = True
#         torch.manual_seed(args.manual_seed)
#         np.random.seed(args.manual_seed)
#         torch.cuda.manual_seed(args.manual_seed)
#         torch.cuda.manual_seed_all(args.manual_seed)
#         random.seed(args.manual_seed)
#         os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
#     main_worker(args)
#
#
# def main_worker(argss):
#     global args
#     args = argss
#     for shot in [5]:
#         save_path = os.path.join('./exp', 'tsk_finetuning', f'exttrain_{args.ext_source}_tskfinetuning_{args.tsk_source}',
#                                  f'{args.model}', f'{args.label_mode}_{shot}', 'model_weight')
#         regis_vis_save_dir = os.path.join('./exp', 'tsk_finetuning', f'exttrain_{args.ext_source}_tskfinetuning_{args.tsk_source}',
#                                           f'{args.model}', f'{args.label_mode}_{shot}', 'guidance_vis_to_draw')
#
#
#         print(args)
#         print("=> creating model ...")
#         print("Classes: {}".format(args.classes))
#
#         ############################# define test set dataloader ##############################
#         test_data = Tsk_Finetune_Dataset(tsk_root=args.tsk_test_root,
#                                          tsk_source=args.tsk_source,
#                                          mode='val',
#                                          transform=None,
#                                          mag=args.mag,
#                                          )
#
#         test_loader = torch.utils.data.DataLoader(test_data, worker_init_fn=worker_init_fn,
#                                                   batch_size=args.batch_size_val,
#                                                   shuffle=False,
#                                                   num_workers=args.workers,
#                                                   pin_memory=True,
#                                                   drop_last=False)
#
#         print(f'{len(test_data)} samples for testing')
#
#         tsk_loader_regis_list = []
#         for tsk_supp_run in [1]:
#             os.makedirs(os.path.join(save_path, f'run_{tsk_supp_run}'), exist_ok=True)
#             print('processing split run: ', tsk_supp_run)
#
#             tsk_supp_data_regis = Tsk_Finetune_Dataset(
#                 mode='train',
#                 tsk_root=args.tsk_train_root,
#                 tsk_source=args.tsk_source,
#                 shot=shot,
#                 run=tsk_supp_run,
#                 transform=None,
#                 mag=args.mag
#             )
#
#
#             # if shot > 10, support sample can not be sent to network restricted to gpu memory
#             if shot < 10:
#                 regis_batch_size = args.tsk_num * shot
#             else:
#                 regis_batch_size = 20
#
#             tsk_supp_loader_regis = torch.utils.data.DataLoader(
#                 tsk_supp_data_regis,
#                 worker_init_fn=worker_init_fn,
#                 batch_size=regis_batch_size,
#                 shuffle=False,
#                 num_workers=args.workers,
#                 pin_memory=True,
#                 sampler=None)
#
#             tsk_loader_regis_list.append(tsk_supp_loader_regis)
#
#             print(f'num of {tsk_supp_run}: {len(tsk_supp_data_regis)}')
#
#         aji_list = []
#         mpq_list = []
#         pq_list = []
#         dice_list = []
#         f1_det_list = []
#         f1_cls_list = []
#         f1_base_list = []
#         f1_novel_list = []
#         multi_run_mean_f_score = np.zeros(shape=(args.classes - 1))
#         for run, _ in enumerate(tsk_loader_regis_list, start=1):
#
#             print(f'################################## start run {run} #####################################')
#
#
#             weight_file = glob.glob(os.path.join(save_path, f'run_{run}/*.pth'))
#             assert len(weight_file) == 1
#             weight_file = weight_file[0]
#
#             model = model_init(model_weight=weight_file)
#
#             print('####################### eval w regis run {} #####################'.format(run))
#
#             aji, F1_det, mpq, F_score, F1_avg, pq, dice = evaluate_one_epoch_patch_level(model,
#                                                                                gened_proto=None,
#                                                                                novel_list=args.novel_list,
#                                                                                base_list=args.base_list,
#                                                                                test_loader=test_loader,
#                                                                                test_dataset=test_data,
#                                                                                num_type=args.tsk_num + 1,
#                                                                                dataset=args.tsk_source,
#                                                                                is_draw=True,
#                                                                                save_dir=regis_vis_save_dir,
#                                                                                num_run=run)
#
#
#             print('aji: {}, mpq: {}, det f1 {}'.format(aji, mpq, F1_det))
#             print('f_score: {}'.format(F_score))
#
#
#             aji_list += [aji]
#             mpq_list += [mpq]
#             f1_det_list += [F1_det]
#             f1_cls_list += [F1_avg]
#             pq_list += [pq]
#             dice_list += [dice]
#             F_score = np.array(F_score).astype(np.float64)
#             base_list = [i - 1 for i in args.base_list]
#             f1_base_list += [np.mean(F_score[base_list])]
#             novel_list = [i - 1 for i in args.novel_list]
#             f1_novel_list += [np.mean(F_score[novel_list])]
#             multi_run_mean_f_score += F_score
#
#         print('aji_mean_std: {}_{}'.format(np.mean(aji_list), np.std(aji_list)))
#         print('pq_mean_std: {}_{}'.format(np.mean(pq_list), np.std(pq_list)))
#         print('dice_mean_std: {}_{}'.format(np.mean(dice_list), np.std(dice_list)))
#         print('detection_f_score_mean_std: {}_{}'.format(np.mean(f1_det_list), np.std(f1_det_list)))
#         print('mpq_mean_std: {}_{}'.format(np.mean(mpq_list), np.std(mpq_list)))
#         print('classification_f_score_mean_std: {}_{}'.format(np.mean(f1_cls_list), np.std(f1_cls_list)))
#         print('classification_f_score_base_mean_std: {}_{}'.format(np.mean(f1_base_list), np.std(f1_base_list)))
#         print('classification_f_score_novel_mean_std: {}_{}'.format(np.mean(f1_novel_list), np.std(f1_novel_list)))
#
#         multi_run_mean_f_score /= len(args.task_run_list)
#         multi_run_mean_f_score = "F1_socre:" + "/".join("{}".format(f) for f in multi_run_mean_f_score)
#         print('mpq_mean_f1: {}'.format(multi_run_mean_f_score))
#
#
# def get_new_proto(val_supp_loader, model, novel_list, base_list, shot):
#     print('>>>>>>>>>>>>>>>> Start New Proto Generation >>>>>>>>>>>>>>>>')
#     model.eval()
#
#     # if shot > 10, support sample can not be sent to network restricted to gpu memory
#     if shot < 10:
#         n_size = args.tsk_num * shot
#     else:
#         n_size = 20
#
#     # new_proto_num_epoch = 1  # 1
#     with torch.no_grad():
#         main_size = 512
#
#         sup_sem_feat_list_bed = []
#         sup_con_feat_list_bed = []
#         sup_cen_feat_list_bed = []
#
#         gened_proto_bed = torch.zeros(args.classes, main_size).cuda()
#
#         # for epoch in range(new_proto_num_epoch):
#         for i, (input, target, _, _, contour_target, centroid_target, _) in enumerate(val_supp_loader):
#             input = input.cuda(non_blocking=True)
#             target = target.cuda(non_blocking=True)
#             contour_target = contour_target.cuda(non_blocking=True)
#             # print('Generating new prototypes {}/{}...'.format(epoch, new_proto_num_epoch))
#             print('base_num: {}, novel_num: {}'.format(len(base_list), len(novel_list)))
#             print('Input: {}, Target: {}.'.format(input.shape, target.shape))
#
#             input = input.contiguous().view(1, n_size, input.size(1), input.size(2), input.size(3))
#             target = target.contiguous().view(1, n_size, target.size(1), target.size(2))
#             contour_target = contour_target.contiguous().view(1, n_size, contour_target.size(1),
#                                                               contour_target.size(2))
#             centroid_target = centroid_target.contiguous().view(1, n_size, centroid_target.size(1),
#                                                             centroid_target.size(2))
#             input = input.repeat(8, 1, 1, 1, 1)
#             target = target.repeat(8, 1, 1, 1)
#             contour_target = contour_target.repeat(8, 1, 1, 1)
#             centroid_target = centroid_target.repeat(8, 1, 1, 1)
#
#             gened_proto, \
#             sup_sem_feat_list, \
#             sup_cen_feat_list, \
#             sup_con_feat_list = model(x=input,
#                                       y_cls=target,
#                                       y_con=contour_target,
#                                       y_cen=centroid_target,
#                                       gen_proto=True,
#                                       novel_list=novel_list,
#                                       base_list=base_list)
#             gened_proto = gened_proto.mean(0)
#             gened_proto_bed = gened_proto_bed + gened_proto
#             # np
#             sup_sem_feat_list_bed += sup_sem_feat_list
#             # cenp
#             sup_cen_feat_list_bed += sup_cen_feat_list
#             # cp
#             sup_con_feat_list_bed += sup_con_feat_list
#
#         gened_proto = gened_proto_bed / len(val_supp_loader)
#         return gened_proto, \
#                sup_sem_feat_list_bed, \
#                sup_cen_feat_list_bed,\
#                sup_con_feat_list_bed
#
#
# def rand_color_visualization(inst_mask):
#     mask = np.zeros(shape=(256, 256, 3))
#
#     ind_list = np.unique(inst_mask).tolist()
#     ind_list.remove(0)
#     for i in ind_list:
#         R = random.randint(0, 255)
#         G = random.randint(0, 255)
#         B = random.randint(0, 255)
#         mask[inst_mask == i] = np.array([B, G, R])
#     return mask
#
# def evaluate_one_epoch_patch_level(model,
#                                    gened_proto,
#                                    novel_list,
#                                    base_list,
#                                    test_loader,
#                                    test_dataset,
#                                    gened_sem_proto=None,
#                                    gened_cen_proto=None,
#                                    gened_con_proto=None,
#                                    num_type=5,
#                                    device='cuda',
#                                    dataset='',
#                                    is_draw=True,
#                                    save_dir='',
#                                    num_run=1):
#
#     eval_with_pt = True if gened_proto != None else False
#     eval_without_pt = not eval_with_pt
#
#     vis_save_dir = save_dir # os.path.join(save_dir, 'vis')
#     os.makedirs(vis_save_dir, exist_ok=True)
#     model.eval()
#     results = []
#     for i, (img_input, cls_target, fore_target, inst_target,
#             contour_target, centroid_target, _) in enumerate(test_loader):
#
#         true_tp = cls_target.numpy()[0]
#         true_inst = inst_target.numpy()[0]
#
#         true_fore_target = fore_target.numpy()[0]
#         contour_target = contour_target.numpy()[0]
#         centroid_target = centroid_target.numpy()[0]
#         ori_img = cv2.imread(test_dataset.img_list[i])
#         img_name = os.path.basename(test_dataset.img_list[i]).split('.')[0]
#         overlay = draw_rand_inst_overlay_scaling(ori_img, true_inst, rand_color=False, draw_center=True)
#
#         label_inst = rand_color_visualization(true_inst)
#         true_sem = cv2.cvtColor(true_fore_target * 255, cv2.COLOR_GRAY2BGR)
#         true_con = cv2.cvtColor(contour_target * 255, cv2.COLOR_GRAY2BGR)
#         true_cen = cv2.cvtColor(centroid_target * 255, cv2.COLOR_GRAY2BGR)
#
#         # pred_vis = draw_rand_inst_overlay_scaling(ori_img, pred_inst, pred_cls, col_dict[dataset])
#         edge1 = np.ones(shape=(256, 10, 3)) * 255
#
#         # vis = np.concatenate([ori_img[48: -48, 48: -48, :], edge1,
#         #                       overlay[48: -48, 48: -48, :], edge1,
#         #                       label_inst[48: -48, 48: -48, :], edge1,
#         #                       true_sem[48: -48, 48: -48, :], edge1,
#         #                       true_con[48: -48, 48: -48, :], edge1,
#         #                       true_cen[48: -48, 48: -48, :]], axis=1)
#
#         vis = np.concatenate([ori_img, edge1,
#                               overlay, edge1,
#                               label_inst, edge1,
#                               true_sem, edge1,
#                               true_con, edge1,
#                               true_cen], axis=1)
#         cv2.imwrite(f'/home/data1/my/dataset/pannuke/Fold_3/inst_gt_vis_0917/{img_name}.png', vis)
#
#
#         # pred_dict_, sem_sim_map, cen_sim_map, con_sim_map = \
#         #     model(x=img_input.to(device),
#         #            eval_model=eval_with_pt,
#         #            eval_model_wo_proto=eval_without_pt,
#         #            eval_with_sg=True,
#         #            gened_proto=gened_proto,
#         #            gened_sem_proto=gened_sem_proto,
#         #            gened_cen_proto=gened_cen_proto,
#         #            gened_con_proto=gened_con_proto,
#         #            novel_list=novel_list,
#         #            base_list=base_list,
#         #            )
#         #
#         # sem_sim_map = sem_sim_map.permute(0, 2, 3, 1).contiguous()
#         # sem_sim_map = F.softmax(sem_sim_map, dim=-1)[..., 1].detach().cpu().numpy()[0]
#         # cen_sim_map = cen_sim_map.permute(0, 2, 3, 1).contiguous()
#         # cen_sim_map = F.softmax(cen_sim_map, dim=-1)[..., 1].detach().cpu().numpy()[0]
#         # con_sim_map = con_sim_map.permute(0, 2, 3, 1).contiguous()
#         # con_sim_map = F.softmax(con_sim_map, dim=-1)[..., 1].detach().cpu().numpy()[0]
#         #
#         # # confirm the branch prediction output order np-cp-tp
#         # pred_dict = OrderedDict()
#         # pred_dict['np'] = pred_dict_['np']
#         # pred_dict['conp'] = pred_dict_['conp']
#         # pred_dict['cenp'] = pred_dict_['cenp']
#         # pred_dict['tp'] = pred_dict_['tp']
#         #
#         # pred_dict = OrderedDict(
#         #     [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
#         # )
#         # pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
#         # pred_dict["conp"] = F.softmax(pred_dict["conp"], dim=-1)[..., 1:]
#         #
#         # if "tp" in pred_dict:
#         #     type_map = F.softmax(pred_dict["tp"], dim=-1)
#         #     type_map = torch.argmax(type_map, dim=-1, keepdim=True)
#         #     type_map = type_map.type(torch.float32)
#         #     pred_dict["tp"] = type_map
#         # pred_output = torch.cat(list(pred_dict.values()), -1)
#         #
#         # # x = pred_output.detach().cpu().numpy()
#         # pred_inst, inst_info_dict = process(pred_output.detach().cpu().numpy()[0],
#         #                                     nr_types=num_type,
#         #                                     sem_guid_map=sem_sim_map,
#         #                                     con_guid_map=con_sim_map,
#         #                                     cen_guid_map=cen_sim_map,
#         #                                     )
#         #
#         # results += [(true_inst[0], true_tp[0],
#         #              pred_output.detach().cpu().numpy()[0],
#         #              pred_inst, inst_info_dict,
#         #              sem_sim_map, cen_sim_map, con_sim_map)]
#
#     ############################ evalute the performence #################################
#     total_aji = 0
#     total_dice = 0
#     list_length = len(results)
#     res = []
#     true_res = []
#     pred_dict = {}
#     true_dict = {}
#     for idx, result in enumerate(results):
#         ori_img = cv2.imread(test_dataset.img_list[idx])
#         img_name = os.path.basename(test_dataset.img_list[idx]).split('.')[0]
#         inst_label, cls_label, pred_output, pred_inst, inst_info_dict, \
#         sem_sim_map, cen_sim_map, con_sim_map = result
#
#         # cls to be ingnored in gt labels (0 means ignore)
#         ambiguous_mask = np.ones_like(inst_label).astype(np.bool)
#
#         if dataset == 'monusac':
#             # order can not be changed
#             ambiguous_mask = (cls_label != 5)
#             inst_label *= ambiguous_mask
#             cls_label *= ambiguous_mask
#
#         pred_cls = np.zeros_like(pred_inst)
#         if inst_info_dict is not None:
#             for inst in range(1, int(np.max(pred_inst)) + 1):
#                 try:
#                     pred_cls[pred_inst == inst] = inst_info_dict[inst]['type']
#                 except:
#                     pass
#
#         # set the cls as 0
#         pred_cls *= ambiguous_mask
#         pred_inst *= ambiguous_mask
#
#         pred_dict[img_name] = generate_cls_info(pred_inst, pred_cls)
#         true_dict[img_name] = generate_cls_info(inst_label, cls_label)
#
#         res += [np.concatenate([pred_inst[..., None], pred_cls[..., None]], axis=-1)[None, ...]]
#         true_res += [np.concatenate([inst_label[..., None], cls_label[..., None]], axis=-1)[None, ...]]
#
#         # draw_overlay
#         if is_draw and num_run == 1:
#
#             pred_np = pred_output[:, :, 0]
#             pred_np = 255 * np.array(pred_np >= 0.5, dtype=np.int32)
#             pred_np = pred_np.astype(np.uint8)
#             pred_np = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
#
#             pred_conp = pred_output[:, :, 1]
#             pred_conp = 255 * np.array(pred_conp >= 0.5, dtype=np.int32)
#             pred_conp = pred_conp.astype(np.uint8)
#             pred_conp = cv2.cvtColor(pred_conp, cv2.COLOR_GRAY2BGR)
#
#             pred_cenp = pred_output[:, :, 2]
#             pred_cenp = 255 * np.array(pred_cenp >= 0.5, dtype=np.int32)
#             pred_cenp = pred_cenp.astype(np.uint8)
#             pred_cenp = cv2.cvtColor(pred_cenp, cv2.COLOR_GRAY2BGR)
#
#             pred_type = pred_output[..., -1]
#             pred_type = draw_overlay_by_index(pred_type, col_dict[dataset])
#
#             # # vis_guidance
#             # sem_guid_map = np.array(sem_sim_map >= 0.5, dtype=np.int32)
#             # con_guid_map = np.array(con_sim_map >= 0.5, dtype=np.int32)
#             # con_guid_map = np.array(cen_sim_map >= 0.5, dtype=np.int32)
#
#             overlay = draw_rand_inst_overlay_scaling(ori_img, pred_inst)
#             pred_inst_show = rand_color_visualization(pred_inst)
#
#             edge1 = np.ones(shape=(160, 6, 3)) * 255
#
#             vis = np.concatenate([overlay[48: -48, 48: -48, :], edge1,
#                                   pred_inst_show[48: -48, 48: -48, :], edge1,
#                                   pred_np[48: -48, 48: -48, :], edge1,
#                                   pred_conp[48: -48, 48: -48, :]], axis=1)
#             cv2.imwrite(f'{vis_save_dir}/{img_name}.png',
#                         vis)
#
#             # cv2.imwrite(
#             #     f'{vis_save_dir}/{img_name}.png',
#             #     pred_vis[48: -48, 48: -48, :])
#
#
#         # if empty test patch, ignore the metric
#         if np.max(inst_label) == 0 and np.max(pred_inst) == 0:
#             aji = 0
#             dice = 0
#             list_length -= 1
#         elif np.max(inst_label) != 0 and np.max(pred_inst) == 0:
#             aji = 0
#             dice = 0
#         else:
#             aji = get_fast_aji(inst_label, pred_inst)
#             dice = get_dice(inst_label, pred_inst)
#
#         total_aji += aji
#         total_dice += dice
#
#     pq_metrics = getmPQ(res, true_res, nr_classes=num_type - 1)
#     pq = pq_metrics['pq']
#     mpq = pq_metrics['multi_pq+'][0]
#     # print("mpq: ", mpq)
#
#     # classification metrics
#     if dataset == 'consep_20x' or dataset == 'monusac_20x' or dataset == 'lizard':
#         _20x = True
#     else:
#         _20x = False
#     results_list = run_nuclei_type_stat(pred_dict, true_dict, _20x=_20x)
#     F1_det = results_list[0]
#     F1_list = results_list[-int(num_type - 1):]
#     F1_avg = np.mean(F1_list)
#
#     return total_aji / list_length, F1_det, mpq, F1_list, F1_avg, pq, total_dice / list_length
#
#
# if __name__ == '__main__':
#     main()


import logging
import os
import random
import cv2
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import glob
import torch.nn.parallel
import torch.optim
import torch.utils.data
from scipy import ndimage
import torch.nn.functional as F
from collections import OrderedDict
from models.post_proc import process
from models.SGFSL import SGFSL
from utils.loss import xentropy_loss, dice_loss, mse_loss
from dataset.nuclei_dataset import Tsk_Finetune_Dataset
from utils import config

from utils.viz_utils import draw_rand_inst_overlay_scaling, draw_overlay_by_index, rand_color_visualization
from utils.metrics import run_nuclei_type_stat
from utils.metrics import generate_cls_info
from utils.metrics import get_fast_aji, getmPQ, get_dice

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

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
            'lizard': lizard_col}

loss_func_dict = {"bce": xentropy_loss,
                  "dice": dice_loss,
                  "con_bce": xentropy_loss,
                  "con_dice": dice_loss,
                  "cen_mse": mse_loss,
                  "type_bce": xentropy_loss,
                  'type_dice': dice_loss}

loss_opts = {"np": {"bce": 1, "dice": 1},
             "conp": {"con_bce": 1, "con_dice": 1},
             "cenp": {"cen_mse": 10},
             "tp": {"type_bce": 1, "type_dice": 1}}


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Nuclei Instance Segmentation')
    parser.add_argument('--config', type=str,
                        # default='./config/tsk_finetuning/ext_consep_tsk_monusac.yaml',
                        # default='./config/tsk_finetuning/ext_pannuke_tsk_monusac.yaml',
                        # default='./config/tsk_finetuning/ext_monusac20x_tsk_lizard.yaml',
                        # default='./config/tsk_finetuning/ext_monusac_tsk_consep.yaml',
                        default='./config/tsk_finetuning/ext_monusac_tsk_pannuke.yaml',
                        help='config file')
    parser.add_argument('opts', help='see .yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)
    np.random.seed(args.manual_seed + worker_id)


def model_init(model_weight):
    model = SGFSL(num_types=args.classes, modal_num=args.modal_num)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.train_gpu)

    ####################################### TODO: load pretrained model #################################
    model_weight = torch.load(model_weight, map_location='cpu')['model']

    load_weight_dict = {k: v for k, v in model_weight.items()
                        if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(load_weight_dict, strict=False)

    return model


def main():
    args = get_parser()
    assert args.classes > 1
    torch.cuda.set_device(args.train_gpu[0])
    global device

    device = torch.device(f'cuda:{args.train_gpu[0]}' if torch.cuda.is_available() else 'cpu')
    print(device)

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
    main_worker(args)


def main_worker(argss):
    global args
    args = argss
    for shot in [5]:
        save_path = os.path.join('./exp', 'tsk_finetuning',
                                 f'exttrain_{args.ext_source}_tskfinetuning_{args.tsk_source}',
                                 f'{args.model}', f'{args.label_mode}_{shot}', 'model_weight')
        regis_vis_save_dir = os.path.join('./exp', 'tsk_finetuning',
                                          f'exttrain_{args.ext_source}_tskfinetuning_{args.tsk_source}',
                                          f'{args.model}', f'{args.label_mode}_{shot}', 'guidance_vis_to_draw')

        print(args)
        print("=> creating model ...")
        print("Classes: {}".format(args.classes))

        ############################# define test set dataloader ##############################
        test_data = Tsk_Finetune_Dataset(tsk_root=args.tsk_test_root,
                                         tsk_source=args.tsk_source,
                                         mode='val',
                                         transform=None,
                                         mag=args.mag,
                                         )

        test_loader = torch.utils.data.DataLoader(test_data, worker_init_fn=worker_init_fn,
                                                  batch_size=args.batch_size_val,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True,
                                                  drop_last=False)

        print(f'{len(test_data)} samples for testing')

        tsk_loader_regis_list = []
        for tsk_supp_run in [1]:
            os.makedirs(os.path.join(save_path, f'run_{tsk_supp_run}'), exist_ok=True)
            print('processing split run: ', tsk_supp_run)

            tsk_supp_data_regis = Tsk_Finetune_Dataset(
                mode='train',
                tsk_root=args.tsk_train_root,
                tsk_source=args.tsk_source,
                shot=shot,
                run=tsk_supp_run,
                transform=None,
                mag=args.mag
            )

            # if shot > 10, support sample can not be sent to network restricted to gpu memory
            if shot < 10:
                regis_batch_size = args.tsk_num * shot
            else:
                regis_batch_size = 20

            tsk_supp_loader_regis = torch.utils.data.DataLoader(
                tsk_supp_data_regis,
                worker_init_fn=worker_init_fn,
                batch_size=regis_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                sampler=None)

            tsk_loader_regis_list.append(tsk_supp_loader_regis)

            print(f'num of {tsk_supp_run}: {len(tsk_supp_data_regis)}')

        aji_list = []
        mpq_list = []
        pq_list = []
        dice_list = []
        f1_det_list = []
        f1_cls_list = []
        f1_base_list = []
        f1_novel_list = []
        multi_run_mean_f_score = np.zeros(shape=(args.classes - 1))
        for run, tsk_loader_regis in enumerate(tsk_loader_regis_list, start=1):
            # ------------------------------------------------------------------------------------------- #
            # weight_file = glob.glob(os.path.join('/home/data1/my/Project/SGFSL/SGFSL_ours/exp/tsk_finetuning/'
            #                                      'exttrain_monusac_20x_tskfinetuning_lizard/SGFSL_a/shot_5/model_weight/',
            #                                      f'run_{run}/*.pth'))
            weight_file = glob.glob(os.path.join('/home/data1/my/Project/SGFSL/SGFSL_ours/exp/tsk_finetuning/'
                                                 'exttrain_monusac_tskfinetuning_pannuke/SGFSL_a/shot_5/model_weight/',
                                                 f'run_{run}/*.pth'))
            assert len(weight_file) == 1
            weight_file = weight_file[0]
            model = model_init(model_weight=weight_file)
            res = evaluate_one_epoch_patch_level_wo_sg(model,
                                                       gened_proto=None,
                                                       novel_list=args.novel_list,
                                                       base_list=args.base_list,
                                                       test_loader=test_loader,
                                                       num_type=args.tsk_num + 1,
                                                       )

            # ------------------------------------------------------------------------------------------- #

            print(f'################################## start run {run} #####################################')

            weight_file = glob.glob(os.path.join(save_path, f'run_{run}/*.pth'))
            assert len(weight_file) == 1
            weight_file = weight_file[0]
            model = model_init(model_weight=weight_file)

            print('####################### eval w regis run {} #####################'.format(run))
            gened_proto, \
                sup_sem_feat_list, \
                sup_cen_feat_list, \
                sup_con_feat_list = get_new_proto(tsk_loader_regis,
                                                  model,
                                                  novel_list=args.novel_list,
                                                  base_list=args.base_list,
                                                  shot=shot)
            gened_sem_proto, \
            gened_cen_proto, \
            gened_con_proto = model.module._sg_bank_registration(sup_sem_feat_list,
                                                                 sup_cen_feat_list,
                                                                 sup_con_feat_list)
            aji, F1_det, mpq, F_score, F1_avg, pq, dice = evaluate_one_epoch_patch_level(model,
                                                                                         gened_proto=None,
                                                                                         gened_sem_proto=gened_sem_proto,
                                                                                         gened_cen_proto=gened_cen_proto,
                                                                                         gened_con_proto=gened_con_proto,
                                                                                         novel_list=args.novel_list,
                                                                                         base_list=args.base_list,
                                                                                         test_loader=test_loader,
                                                                                         test_dataset=test_data,
                                                                                         num_type=args.tsk_num + 1,
                                                                                         dataset=args.tsk_source,
                                                                                         res_wo_sg=res,
                                                                                         is_draw=True,
                                                                                         save_dir=regis_vis_save_dir,
                                                                                         num_run=run)
            print('aji: {}, mpq: {}, det f1 {}'.format(aji, mpq, F1_det))
            print('f_score: {}'.format(F_score))
            aji_list += [aji]
            mpq_list += [mpq]
            f1_det_list += [F1_det]
            f1_cls_list += [F1_avg]
            pq_list += [pq]
            dice_list += [dice]
            F_score = np.array(F_score).astype(np.float64)
            base_list = [i - 1 for i in args.base_list]
            f1_base_list += [np.mean(F_score[base_list])]
            novel_list = [i - 1 for i in args.novel_list]
            f1_novel_list += [np.mean(F_score[novel_list])]
            multi_run_mean_f_score += F_score

        print('aji_mean_std: {}_{}'.format(np.mean(aji_list), np.std(aji_list)))
        print('pq_mean_std: {}_{}'.format(np.mean(pq_list), np.std(pq_list)))
        print('dice_mean_std: {}_{}'.format(np.mean(dice_list), np.std(dice_list)))
        print('detection_f_score_mean_std: {}_{}'.format(np.mean(f1_det_list), np.std(f1_det_list)))
        print('mpq_mean_std: {}_{}'.format(np.mean(mpq_list), np.std(mpq_list)))
        print('classification_f_score_mean_std: {}_{}'.format(np.mean(f1_cls_list), np.std(f1_cls_list)))
        print('classification_f_score_base_mean_std: {}_{}'.format(np.mean(f1_base_list), np.std(f1_base_list)))
        print('classification_f_score_novel_mean_std: {}_{}'.format(np.mean(f1_novel_list), np.std(f1_novel_list)))

        multi_run_mean_f_score /= len(args.task_run_list)
        multi_run_mean_f_score = "F1_socre:" + "/".join("{}".format(f) for f in multi_run_mean_f_score)
        print('mpq_mean_f1: {}'.format(multi_run_mean_f_score))


def get_new_proto(val_supp_loader, model, novel_list, base_list, shot):
    print('>>>>>>>>>>>>>>>> Start New Proto Generation >>>>>>>>>>>>>>>>')
    model.eval()

    # if shot > 10, support sample can not be sent to network restricted to gpu memory
    if shot < 10:
        n_size = args.tsk_num * shot
    else:
        n_size = 20

    # new_proto_num_epoch = 1  # 1
    with torch.no_grad():
        main_size = 512

        sup_sem_feat_list_bed = []
        sup_con_feat_list_bed = []
        sup_cen_feat_list_bed = []

        gened_proto_bed = torch.zeros(args.classes, main_size).cuda()

        # for epoch in range(new_proto_num_epoch):
        for i, (input, target, _, _, contour_target, centroid_target, _) in enumerate(val_supp_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            contour_target = contour_target.cuda(non_blocking=True)
            # print('Generating new prototypes {}/{}...'.format(epoch, new_proto_num_epoch))
            print('base_num: {}, novel_num: {}'.format(len(base_list), len(novel_list)))
            print('Input: {}, Target: {}.'.format(input.shape, target.shape))

            input = input.contiguous().view(1, n_size, input.size(1), input.size(2), input.size(3))
            target = target.contiguous().view(1, n_size, target.size(1), target.size(2))
            contour_target = contour_target.contiguous().view(1, n_size, contour_target.size(1),
                                                              contour_target.size(2))
            centroid_target = centroid_target.contiguous().view(1, n_size, centroid_target.size(1),
                                                                centroid_target.size(2))
            input = input.repeat(8, 1, 1, 1, 1)
            target = target.repeat(8, 1, 1, 1)
            contour_target = contour_target.repeat(8, 1, 1, 1)
            centroid_target = centroid_target.repeat(8, 1, 1, 1)

            gened_proto, \
                sup_sem_feat_list, \
                sup_cen_feat_list, \
                sup_con_feat_list = model(x=input,
                                          y_cls=target,
                                          y_con=contour_target,
                                          y_cen=centroid_target,
                                          gen_proto=True,
                                          novel_list=novel_list,
                                          base_list=base_list)
            gened_proto = gened_proto.mean(0)
            gened_proto_bed = gened_proto_bed + gened_proto
            # np
            sup_sem_feat_list_bed += sup_sem_feat_list
            # cenp
            sup_cen_feat_list_bed += sup_cen_feat_list
            # cp
            sup_con_feat_list_bed += sup_con_feat_list

        gened_proto = gened_proto_bed / len(val_supp_loader)
        return gened_proto, \
            sup_sem_feat_list_bed, \
            sup_cen_feat_list_bed, \
            sup_con_feat_list_bed


def rand_color_visualization(inst_mask):
    mask = np.zeros(shape=(256, 256, 3))

    ind_list = np.unique(inst_mask).tolist()
    ind_list.remove(0)
    for i in ind_list:
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        mask[inst_mask == i] = np.array([B, G, R])
    return mask


def evaluate_one_epoch_patch_level(model,
                                   gened_proto,
                                   novel_list,
                                   base_list,
                                   test_loader,
                                   test_dataset,
                                   res_wo_sg,
                                   gened_sem_proto=None,
                                   gened_cen_proto=None,
                                   gened_con_proto=None,
                                   num_type=5,
                                   device='cuda',
                                   dataset='',
                                   is_draw=True,
                                   save_dir='',
                                   num_run=1):
    eval_with_pt = True if gened_proto != None else False
    eval_without_pt = not eval_with_pt

    vis_save_dir = save_dir  # os.path.join(save_dir, 'vis')
    os.makedirs(vis_save_dir, exist_ok=True)
    model.eval()
    results = []
    for i, (img_input, cls_target, fore_target, inst_target,
            contour_target, centroid_target, _) in enumerate(test_loader):

        true_tp = cls_target.numpy()
        true_inst = inst_target.numpy()

        pred_dict_, sem_sim_map, cen_sim_map, con_sim_map = \
            model(x=img_input.to(device),
                  eval_model=eval_with_pt,
                  eval_model_wo_proto=eval_without_pt,
                  eval_with_sg=True,
                  gened_proto=gened_proto,
                  gened_sem_proto=gened_sem_proto,
                  gened_cen_proto=gened_cen_proto,
                  gened_con_proto=gened_con_proto,
                  novel_list=novel_list,
                  base_list=base_list,
                  )

        sem_sim_map = sem_sim_map.permute(0, 2, 3, 1).contiguous()
        sem_sim_map = F.softmax(sem_sim_map, dim=-1)[..., 1].detach().cpu().numpy()[0]
        cen_sim_map = cen_sim_map.permute(0, 2, 3, 1).contiguous()
        cen_sim_map = F.softmax(cen_sim_map, dim=-1)[..., 1].detach().cpu().numpy()[0]
        con_sim_map = con_sim_map.permute(0, 2, 3, 1).contiguous()
        con_sim_map = F.softmax(con_sim_map, dim=-1)[..., 1].detach().cpu().numpy()[0]

        # # confirm the branch prediction output order np-cp-tp
        # pred_dict = OrderedDict()
        # pred_dict['np'] = pred_dict_['np']
        # pred_dict['conp'] = pred_dict_['conp']
        # pred_dict['cenp'] = pred_dict_['cenp']
        # pred_dict['tp'] = pred_dict_['tp']
        #
        # pred_dict = OrderedDict(
        #     [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        # )
        # pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        # pred_dict["conp"] = F.softmax(pred_dict["conp"], dim=-1)[..., 1:]
        #
        # if "tp" in pred_dict:
        #     type_map = F.softmax(pred_dict["tp"], dim=-1)
        #     type_map = torch.argmax(type_map, dim=-1, keepdim=True)
        #     type_map = type_map.type(torch.float32)
        #     pred_dict["tp"] = type_map
        # pred_output = torch.cat(list(pred_dict.values()), -1)
        pred_output = res_wo_sg[i][0]
        pred_inst_wo_sg = res_wo_sg[i][1]

        # x = pred_output.detach().cpu().numpy()
        pred_inst, inst_info_dict = process(pred_output,
                                            nr_types=num_type,
                                            sem_guid_map=sem_sim_map,
                                            con_guid_map=con_sim_map,
                                            cen_guid_map=cen_sim_map,
                                            )

        results += [(true_inst[0], true_tp[0],
                     pred_output,
                     pred_inst_wo_sg,
                     pred_inst, inst_info_dict,
                     sem_sim_map, cen_sim_map, con_sim_map)]

    ############################ evalute the performence #################################
    total_aji = 0
    total_dice = 0
    list_length = len(results)
    res = []
    true_res = []
    pred_dict = {}
    true_dict = {}
    for idx, result in enumerate(results):
        ori_img = cv2.imread(test_dataset.img_list[idx])
        img_name = os.path.basename(test_dataset.img_list[idx]).split('.')[0]
        inst_label, cls_label, pred_output, pred_inst_wo_sg, \
            pred_inst, inst_info_dict, \
            sem_sim_map, cen_sim_map, con_sim_map = result

        # cls to be ingnored in gt labels (0 means ignore)
        ambiguous_mask = np.ones_like(inst_label).astype(np.bool)

        if dataset == 'monusac':
            # order can not be changed
            ambiguous_mask = (cls_label != 5)
            inst_label *= ambiguous_mask
            cls_label *= ambiguous_mask

        pred_cls = np.zeros_like(pred_inst)
        if inst_info_dict is not None:
            for inst in range(1, int(np.max(pred_inst)) + 1):
                try:
                    pred_cls[pred_inst == inst] = inst_info_dict[inst]['type']
                except:
                    pass

        # set the cls as 0
        pred_cls *= ambiguous_mask
        pred_inst *= ambiguous_mask

        pred_dict[img_name] = generate_cls_info(pred_inst, pred_cls)
        true_dict[img_name] = generate_cls_info(inst_label, cls_label)

        res += [np.concatenate([pred_inst[..., None], pred_cls[..., None]], axis=-1)[None, ...]]
        true_res += [np.concatenate([inst_label[..., None], cls_label[..., None]], axis=-1)[None, ...]]

        # draw_overlay
        if is_draw and num_run == 1:
            # gt_vis = draw_overlay_scaling(ori_img, inst_label, cls_label, col_dict[dataset])
            # pred_vis = draw_overlay_scaling(ori_img, pred_inst, pred_cls, col_dict[dataset])

            pred_np = pred_output[:, :, 0]
            pred_np = np.array(pred_np >= 0.5, dtype=np.int32)
            pred_conp = pred_output[:, :, 1]
            pred_conp = np.array(pred_conp >= 0.5, dtype=np.int32)
            pred_cenp = pred_output[:, :, 2]
            pred_cenp = np.array(pred_cenp >= 0.5, dtype=np.int32)

            # pred_type = pred_output[..., -1]
            # pred_type = draw_overlay_by_index(pred_type, col_dict[dataset])
            # vis_guidance
            sem_guid_map = np.array(sem_sim_map >= 0.5, dtype=np.int32)
            con_guid_map = np.array(con_sim_map >= 0.5, dtype=np.int32)
            cen_guid_map = np.array(cen_sim_map >= 0.5, dtype=np.int32)

            pred_np[(pred_np == 0) & (sem_guid_map != 0)] = 2
            pred_conp[(pred_conp == 0) & (con_guid_map != 0)] = 2
            pred_cenp[(pred_cenp == 0) & (cen_guid_map != 0)] = 2
            pred_np = draw_overlay_by_index(pred_np, np.array([[0, 0, 0],
                                                               [255, 255, 255],
                                                               [0, 0, 255]]))
            pred_conp = draw_overlay_by_index(pred_conp, np.array([[0, 0, 0],
                                                                   [255, 255, 255],
                                                                   [0, 0, 255]]))
            pred_cenp = draw_overlay_by_index(pred_cenp, np.array([[0, 0, 0],
                                                                   [255, 255, 255],
                                                                   [0, 0, 255]]))


            true_overlay = draw_rand_inst_overlay_scaling(ori_img, inst_label,
                                                          rand_color=False, draw_center=True)
            pred_overlay = draw_rand_inst_overlay_scaling(ori_img, pred_inst_wo_sg,
                                                          rand_color=False, draw_center=True)
            sg_pred_overlay = draw_rand_inst_overlay_scaling(ori_img, pred_inst,
                                                             rand_color=False, draw_center=True)
            # pred_inst_show = rand_color_visualization(pred_inst)

            edge1 = np.ones(shape=(256, 10, 3)) * 255

            # vis = np.concatenate([ori_img[48: -48, 48: -48, :], edge1,
            #                       true_overlay[48: -48, 48: -48, :], edge1,
            #                       pred_np[48: -48, 48: -48, :], edge1,
            #                       pred_conp[48: -48, 48: -48, :], edge1,
            #                       pred_cenp[48: -48, 48: -48, :], edge1,
            #                       pred_overlay[48: -48, 48: -48, :], edge1,
            #                       sg_pred_overlay[48: -48, 48: -48, :]
            #                       ], axis=1)
            vis = np.concatenate([ori_img, edge1,
                                  true_overlay, edge1,
                                  pred_np, edge1,
                                  pred_conp, edge1,
                                  pred_cenp, edge1,
                                  pred_overlay, edge1,
                                  sg_pred_overlay
                                  ], axis=1)

            cv2.imwrite(f'{vis_save_dir}/{img_name}.png',
                        vis)

            # overlay = draw_overlay_scaling(ori_img, inst_label, cls_label, col_dict[dataset])
            # pred_vis = draw_overlay_scaling(ori_img, pred_inst, pred_cls, col_dict[dataset])

            # cv2.imwrite(
            #     f'{vis_save_dir}/{img_name}.png',
            #     pred_vis[48: -48, 48: -48, :])

        # if empty test patch, ignore the metric
        if np.max(inst_label) == 0 and np.max(pred_inst) == 0:
            aji = 0
            dice = 0
            list_length -= 1
        elif np.max(inst_label) != 0 and np.max(pred_inst) == 0:
            aji = 0
            dice = 0
        else:
            aji = get_fast_aji(inst_label, pred_inst)
            dice = get_dice(inst_label, pred_inst)

        total_aji += aji
        total_dice += dice

    pq_metrics = getmPQ(res, true_res, nr_classes=num_type - 1)
    pq = pq_metrics['pq']
    mpq = pq_metrics['multi_pq+'][0]
    # print("mpq: ", mpq)

    # classification metrics
    if dataset == 'consep_20x' or dataset == 'monusac_20x' or dataset == 'lizard':
        _20x = True
    else:
        _20x = False
    results_list = run_nuclei_type_stat(pred_dict, true_dict, _20x=_20x)
    F1_det = results_list[0]
    F1_list = results_list[-int(num_type - 1):]
    F1_avg = np.mean(F1_list)

    return total_aji / list_length, F1_det, mpq, F1_list, F1_avg, pq, total_dice / list_length


def evaluate_one_epoch_patch_level_wo_sg(model,
                                   gened_proto,
                                   novel_list,
                                   base_list,
                                   test_loader,
                                   num_type=5,
                                   device='cuda',
                                   ):
    eval_with_pt = True if gened_proto != None else False
    eval_without_pt = not eval_with_pt


    model.eval()
    results = []
    for i, (img_input, cls_target, fore_target, inst_target,
            contour_target, centroid_target, _) in enumerate(test_loader):

        pred_dict_, sem_sim_map, cen_sim_map, con_sim_map = \
            model(x=img_input.to(device),
                  eval_model=eval_with_pt,
                  eval_model_wo_proto=eval_without_pt,
                  eval_with_sg=True,
                  gened_proto=gened_proto,
                  novel_list=novel_list,
                  base_list=base_list,
                  )


        # confirm the branch prediction output order np-cp-tp
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
        pred_output = torch.cat(list(pred_dict.values()), -1)

        pred_inst, inst_info_dict = process(pred_output.detach().cpu().numpy()[0],
                                            nr_types=num_type)
        results += [(pred_output.detach().cpu().numpy()[0],  pred_inst)]

    return results

if __name__ == '__main__':
    main()

