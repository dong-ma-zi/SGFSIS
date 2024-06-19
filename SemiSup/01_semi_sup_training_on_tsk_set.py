import os
import torch
import torch.multiprocessing
import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')
import glob
import numpy as np
import argparse
import pickle
from models.MTQuadNet.net_desc import QuadNet
from itertools import cycle
from utils.logger import get_logger
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from dataset.dataset_monusac import Dataset_labeled, Base_Dataset, train_split_shot, Dataset_unlabeled
from dataset.dataset_monusac import QuadCompose, QuadRandomRotate, QuadRandomFlip
import torch.nn.functional as F
from collections import OrderedDict
from utils.loss import dice_loss, mse_loss, xentropy_loss
from models.QuadBranchNet.post_proc import process
import cv2
from utils.metrics import run_nuclei_type_stat
from utils.metrics import generate_cls_info
from utils.metrics import get_fast_aji, getmPQ

np.random.seed(16)
torch.manual_seed(16)
torch.cuda.manual_seed(16)
torch.cuda.manual_seed_all(16)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='mean Teacher model')
parser.add_argument('--dataset', type=str, default='monusac', help='monusac lizard...')
parser.add_argument('--mag', type=str, default='40x')

parser.add_argument('--sup_mode',
                    type=str,
                    default='Shot',
                    help="Shot | Ratio")

parser.add_argument('--pretrained_weights',
                    type=str,
                    default='/home/data1/my/Project/SGFSL/FullSup/exp/',
                    help="model pretrained on few labeled dataset")
parser.add_argument('--save_dir', type=str, default='./exp')

parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--l_batch_size', type=int, default=4)
parser.add_argument('--ul_batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=5e-6) # 5e-5 for consep 1e-5 for pannuke 5e-5
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--warm_up_epoch', type=int, default=1, help='warm up epoch for mean teacher')
parser.add_argument('--epoch', type=int, default=200, help='training epoch for mean teacher')
parser.add_argument('--save_per_epoch', type=int, default=40, help='save model per epoch')
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--shot', type=float, default=20, help='shots for labeled shot')
parser.add_argument('--num_type', type=int, default=5, help='nuclei types include background')
# parser.add_argument('--novel_list', type=list, default=[4, 5, 6], help="")
# parser.add_argument('--base_list', type=list, default=[1, 2, 3], help="")
parser.add_argument('--novel_list', type=list, default=[3, 4], help="")
parser.add_argument('--base_list', type=list, default=[1, 2], help="")
# parser.add_argument('--novel_list', type=list, default=[1, 4], help="")
# parser.add_argument('--base_list', type=list, default=[2, 3], help="")

parser.add_argument('--train_root', type=str,
                    default='/home/data1/my/dataset/monusac/extracted_mirror/train/256x256_256x256/',
                    # default='/home/data1/my/dataset/consep/extracted_mirror/train/256x256_128x128/',
                    # default='/home/data1/my/dataset/lizard/Train/',
                    help="data root for training and testing data")
parser.add_argument('--test_root', type=str,
                    default='/home/data1/my/dataset/monusac/extracted_mirror/test/256x256_256x256/',
                    # default='/home/data1/my/dataset/consep/extracted_mirror/valid/256x256_256x256/',
                    # default='/home/data1/my/dataset/lizard/Test/',
                    help="data root for training and testing data")

args = parser.parse_args()


pannuke_col = np.array([[0, 0, 255],
                        [0, 255, 0],
                        [255, 0, 0],
                        [0, 255, 255],
                        [0, 165, 255]])

consep_col = np.array([[0, 255, 255],
                       [255, 0, 255],
                       [0, 0, 255],
                       [255, 0, 0]])

lizard_col = np.array([[0, 165, 255],
                       [0, 255, 0],
                       [0, 0, 255],
                       [255, 255, 0],
                       [255, 0, 0],
                       [0, 255, 255]])

monusac_col = np.array([[0, 0, 255],
                        [0, 255, 255],
                        [0, 255, 0],
                        [255, 0, 0]])

loss_func_dict = {"bce": xentropy_loss,
                  "dice": dice_loss,
                  "con_bce": xentropy_loss,
                  "con_dice": dice_loss,
                  "cen_mse": mse_loss,
                  "type_bce": xentropy_loss,
                  "type_dice": dice_loss,
                  "un_np": mse_loss,
                  "un_conp": mse_loss,
                  "un_cenp": mse_loss,
                  "un_tp": mse_loss}


loss_config = {"np": {"bce": 1, "dice": 1},
               "conp": {"con_bce": 1, "con_dice": 1},
               "cenp": {"cen_mse": 10},
               "tp": {"type_bce": 1, "type_dice": 1}}

unsup_loss_config = {"np": {"un_np": 1},
                     "conp": {"un_conp": 1},
                     "cenp": {"un_cenp": 1},
                     "tp": {"un_tp": 1}}


def load_pretrained_weight(model, weight_path=None): # , device):
    # model = nn.DataParallel(model).cuda()
    if weight_path != None:
        assert os.path.exists(weight_path)
        weight = torch.load(weight_path)['model']
        model.load_state_dict(weight, strict=True)
    return model

class Trainer:
    def __init__(self, teacher, student, num_types, run,
                 supervised_loader, unsupervised_loader,
                 valid_dataset, valid_loader=None):

        self.teacher = teacher
        self.student = student

        self.run = run
        self.num_types = num_types
        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.valid_dataset = valid_dataset
        self.valid_loader = valid_loader

        self.optimizer = torch.optim.Adam(self.student.module.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
        #                                                     step_size=args.lr_scheduler, gamma=0.1)

        self.mode = 'semi'
        self.save_per_epoch = args.save_per_epoch
        self.output_dir = args.save_dir

    def update_teachers(self, alpha=0.999):
        with torch.no_grad():
            for param_teacher, param_student in zip(self.teacher.module.parameters(),
                                                    self.student.module.parameters()):
                param_teacher.data.mul_(alpha).add_(param_student.data, alpha=1-alpha)


    def _train_epoch(self, num_type):

        assert self.mode == 'semi'
        accu_loss = {"sup": {"bce": 0,
                             "dice": 0,
                             "con_bce": 0,
                             "con_dice": 0,
                             "cen_mse": 0,
                             "type_bce": 0,
                             "type_dice": 0},

                     "unsup": {"un_np": 0,
                               "un_conp": 0,
                               "un_cenp": 0,
                               "un_tp": 0},
                     "total": 0}

        self.teacher.eval()
        self.student.eval()

        for _, (data_dict_l, data_dict_ul) in enumerate(zip(self.supervised_loader, self.unsupervised_loader)):
            if self.mode == 'semi':
                # data_dict_l, data_dict_ul = next(dataloader)
                img_l = data_dict_l['img'].cuda().type(torch.float32)
                # convert np , contour and type labels to one hot format
                np_map_l = data_dict_l["np_map"].cuda().type(torch.int64)
                np_map_l = F.one_hot(np_map_l, num_classes=2).type(torch.float32)
                con_l = data_dict_l["con_map"].cuda().type(torch.int64)
                con_l = F.one_hot(con_l, num_classes=2).type(torch.float32)
                tp_l = data_dict_l["tp_map"].cuda().type(torch.int64)
                tp_l = F.one_hot(tp_l, num_classes=num_type).type(torch.float32)

                cen_l = data_dict_l["gau_map"].cuda().type(torch.float32)

                # unlabeled data
                img_ul_wk = data_dict_ul['img_u_w'].cuda().type(torch.float32)
                img_ul_st = data_dict_ul['img_u_s'].cuda().type(torch.float32)

                true_dict_l = {
                    "np": np_map_l,
                    "conp": con_l,
                    "cenp": cen_l,
                    "tp": tp_l
                }

                ################################## student branch ########################
                ## Forward pass on labeled data in student model
                output_labeled_student = self.student(img_l)
                output_labeled_student = OrderedDict(
                    [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in output_labeled_student.items()]
                )
                output_labeled_student["np"] = F.softmax(output_labeled_student["np"], dim=-1)
                output_labeled_student["tp"] = F.softmax(output_labeled_student["tp"], dim=-1)
                output_labeled_student["conp"] = F.softmax(output_labeled_student["conp"], dim=-1)
                output_labeled_student["cenp"] = output_labeled_student["cenp"][:, :, :, 0]

                # cal loss for labeled data
                loss_sup = 0
                for branch_name in output_labeled_student.keys():
                    for loss_name, loss_weight in loss_config[branch_name].items():
                        loss_func = loss_func_dict[loss_name]
                        loss_args = [true_dict_l[branch_name], output_labeled_student[branch_name]]
                        if loss_name == "msge":
                            loss_args.append(true_dict_l["np"][..., 1])
                        term_loss = loss_func(*loss_args) * loss_weight
                        accu_loss["sup"][loss_name] += term_loss.cpu().item()
                        loss_sup += term_loss

                ################################## teacher branch ########################
                ## Forward pass on unlabeled data
                with torch.no_grad():
                    # self.teacher.eval()  ## make unsupervised loss do not converge
                    output_unlabeled_teacher = self.teacher(img_ul_wk)
                    output_unlabeled_teacher = OrderedDict(
                        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in output_unlabeled_teacher.items()]
                    )
                output_unlabeled_student = self.student(img_ul_st)
                output_unlabeled_student = OrderedDict(
                    [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in output_unlabeled_student.items()]
                )

                # cal loss for unlabeled data
                loss_unsup = 0
                for branch_name in output_unlabeled_student.keys():
                    for loss_name, loss_weight in unsup_loss_config[branch_name].items():
                        loss_func = loss_func_dict[loss_name]
                        # consistence restriction between labeled data and unlabeled data
                        loss_args = [output_unlabeled_teacher[branch_name], output_unlabeled_student[branch_name]]
                        term_loss = loss_func(*loss_args) * loss_weight
                        accu_loss["unsup"][loss_name] += term_loss.cpu().item()
                        loss_unsup += term_loss

                # sum the loss value of sup and unsup data
                loss_total = loss_sup + loss_unsup
                accu_loss["total"] += loss_total.cpu().item()

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()
                self.update_teachers(alpha=0.999)
        return accu_loss

    def evaluate_one_epoch_patch_level(self,
                                       model,
                                       test_loader,
                                       test_dataset,
                                       num_type=5,
                                       device='cuda',
                                       dataset='',
                                       is_draw=True,
                                       num_run=1,
                                       save_dir=''):
        vis_save_dir = os.path.join(save_dir, 'test_patch_visualization')
        os.makedirs(vis_save_dir, exist_ok=True)
        model.eval()

        results = []
        for data in test_loader:

            img_l = data["img"].to(device).type(torch.float32)
            true_tp = data["tp_map"].numpy()
            true_inst = data["inst_map"].numpy()

            # cal the prediction results
            with torch.no_grad():
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
            # ori_img = cv2.imread(test_dataset.img_path_list[idx])
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

            # if is_draw and num_run == 1:
            #     pred_np = pred_output[:, :, 0]
            #     pred_np = 255 * np.array(pred_np >= 0.5, dtype=np.int32)
            #     pred_np = pred_np.astype(np.uint8)
            #     pred_np = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
            #
            #     pred_conp = pred_output[:, :, 1]
            #     pred_conp = 255 * np.array(pred_conp >= 0.5, dtype=np.int32)
            #     pred_conp = pred_conp.astype(np.uint8)
            #     pred_conp = cv2.cvtColor(pred_conp, cv2.COLOR_GRAY2BGR)
            #
            #     pred_cenp = pred_output[:, :, 2]
            #     pred_cenp = 255 * np.array(pred_cenp >= 0.5, dtype=np.int32)
            #     pred_cenp = pred_cenp.astype(np.uint8)
            #     pred_cenp = cv2.cvtColor(pred_cenp, cv2.COLOR_GRAY2BGR)
            #
            #     pred_type = pred_output[..., -1]
            #     pred_type = draw_overlay_by_index(pred_type, col_dict[dataset])
            #
            #     overlay = cv2.imread(
            #         f'/home/data1/my/dataset/monusac/extracted_mirror/test/256x256_256x256/overlay/{img_name}.png')
            #     pred_vis = draw_overlay_scaling(ori_img, pred_inst, pred_cls, col_dict[dataset])
            #
            #     edge1 = np.ones(shape=(pred_vis.shape[0], 5, 3)) * 255
            #     vis = np.concatenate([overlay, edge1,
            #                           pred_vis, edge1,
            #                           pred_np, edge1,
            #                           pred_conp, edge1,
            #                           pred_cenp, edge1,
            #                           pred_type], axis=1)
            #     cv2.imwrite(f'{vis_save_dir}/{img_name}.png', vis)
            #     # gt_vis = draw_overlay_scaling(ori_img, inst_label, cls_label, col_dict[dataset])
            #     # cv2.imwrite(f'/home/data1/my/dataset/consep/extracted_mirror/valid/256x256_256x256/overlay/{img_name}.png', gt_vis)

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

        pq_metrics = getmPQ(res, true_res, nr_classes=num_type - 1)
        mpq = pq_metrics['multi_pq+'][0]
        print("mpq: ", mpq)

        # classification metrics
        if dataset == 'consep_20x' or dataset == 'monusac_20x' or dataset == 'lizard':
            _20x = True
        else:
            _20x = False

        results_list = run_nuclei_type_stat(pred_dict, true_dict, _20x=_20x)
        F1_det = results_list[0]
        F1_list = results_list[-int(num_type - 1):]
        F1_avg = np.mean(F1_list)

        return total_aji / list_length, F1_det, mpq, F1_list, F1_avg

    def train(self):
        total_l = len(supervised_dataloader)
        total_ul = len(unsupervised_dataloader)
        for epoch in range(args.warm_up_epoch, args.epoch + 1):
            # logger.info('Training Epoch %d =======================================>' % (epoch))
            accu_loss = self._train_epoch(self.num_types)
            logger.info(
                "EPOCH{}  loss:{:.3f}  bce:{:.3f}  dice:{:.3f} con_bce:{:.3f}  con_dice:{:.3f} "
                "cen_mse:{:.3f} type_bce:{:.3f} type_dice:{:.3f} " \
                "un_np:{:.6f} un_conp:{:.6f} un_cenp:{:.6f} un_tp:{:.6f}".format(
                    epoch, accu_loss["total"] / total_l,
                    accu_loss["sup"]["bce"] / total_l,
                    accu_loss["sup"]["dice"] / total_l,
                    accu_loss["sup"]["con_bce"] / total_l,
                    accu_loss["sup"]["con_dice"] / total_l,
                    accu_loss["sup"]["cen_mse"] / total_l,
                    accu_loss["sup"]["type_bce"] / total_l,
                    accu_loss["sup"]["type_dice"] / total_l,
                    accu_loss["unsup"]["un_np"] / total_ul,
                    accu_loss["unsup"]["un_conp"] / total_ul,
                    accu_loss["unsup"]["un_cenp"] / total_ul,
                    accu_loss["unsup"]["un_tp"] / total_ul,
                )
            )

            ############################# validation ############################
            if epoch % self.save_per_epoch == 0:
                aji, F1_det, mpq, F_score, F1_avg = self.evaluate_one_epoch_patch_level(
                    model=self.teacher,
                    test_loader=self.valid_loader,
                    test_dataset=self.valid_dataset,
                    num_type=args.num_type,
                    device=args.device,
                    dataset=args.dataset,
                    is_draw=False,
                    num_run=self.run,
                    save_dir=vis_save_dir)

                logger.info('\n\n')
                logger.info('Epoch: {}, aji: {}, pq: {}, F_det: {}'.format(epoch, aji, mpq, F1_det))
                logger.info('Epoch: {}, f_score: {}'.format(epoch, F_score))

                # if epoch % args.epoch == 0:
                save_files = {
                    'model': self.teacher.state_dict(),
                    'epoch': epoch}
                torch.save(save_files,
                           "{}/run_{}/model_{}_aji_{}_mpq_{}.pth".format(args.save_dir, self.run, epoch, aji, mpq))


        return aji, F1_det, mpq, F_score, F1_avg


if __name__ == '__main__':
    args.save_dir = os.path.join(args.save_dir, args.dataset, f'labeledShot_{args.shot}')
    os.makedirs(args.save_dir, exist_ok=True)
    vis_save_dir = os.path.join(args.save_dir, 'vis')
    os.makedirs(vis_save_dir, exist_ok=True)
    import datetime
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    logger = get_logger(os.path.join(args.save_dir, f'train_log_{cur_time}.txt'))
    logger.info(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ########################### Initialize the semi-sup data loader ########################
    labeled_transforms = QuadCompose([QuadRandomRotate(),
                                      QuadRandomFlip()])

    assert os.path.exists(f'../rand_split_file_{args.dataset}/labeled_shot_{args.shot}')

    # repeat the exp args.runs times
    print('num_run:', args.runs)
    global aji_list, mpq_list, multi_run_mean_f_score
    aji_list = []
    mpq_list = []
    f1_det_list = []
    f1_cls_list = []
    f1_base_list = []
    f1_novel_list = []
    multi_run_mean_f_score = np.zeros(shape=(args.num_type - 1))

    for run in range(1, args.runs + 1):
    # for run in range(5, 6):
        ############################################ model initial ########################################
        pretrained_weight = glob.glob(os.path.join(args.pretrained_weights, args.dataset,
                                                   f'labeledShot_{args.shot}/run_{run}/*.pth'))
        assert len(pretrained_weight) == 1 # load pretrained weights
        pretrained_weight = pretrained_weight[0]
        teacher = load_pretrained_weight(QuadNet(num_types=args.num_type), pretrained_weight)
        student = load_pretrained_weight(QuadNet(num_types=args.num_type), pretrained_weight)
        teacher = nn.DataParallel(teacher).cuda()
        student = nn.DataParallel(student).cuda()

        logger.info(f'#####################################run_{run}##########################################')
        os.makedirs(os.path.join(args.save_dir, f'run_{run}'),
                    exist_ok=True)

        with open(f'../rand_split_file_{args.dataset}/labeled_{args.sup_mode.lower()}_{args.shot}/split_{run}.pkl', 'rb') as f:
            split = pickle.load(f)
        labeled_idx = split['labeled']
        unlabeled_idx = split['unlabeled']


        train_root = args.train_root
        test_root = args.test_root

        dataset_labeled = Dataset_labeled(data_root=train_root,
                                          idx=labeled_idx,
                                          transforms=labeled_transforms,
                                          mag=args.mag)
        dataset_unlabeled = Dataset_unlabeled(data_root=train_root,
                                              idx=unlabeled_idx)
        dataset_test = Dataset_labeled(data_root=test_root,
                                       idx=[],
                                       mag=args.mag)

        logger.info(f'{len(dataset_labeled)} labeled {len(dataset_unlabeled)} unlabeled data for training')

        supervised_dataloader = DataLoader(dataset=dataset_labeled,
                                    batch_size=args.l_batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    drop_last=True)
        unsupervised_dataloader = DataLoader(dataset=dataset_unlabeled,
                                      batch_size=args.ul_batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      drop_last=True)
        loader_test = DataLoader(dataset=dataset_test,
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 drop_last=False)

        trainer = Trainer(teacher, student, args.num_type, run,
                          supervised_dataloader,
                          unsupervised_dataloader,
                          dataset_test,
                          loader_test)

        aji, F1_det, mpq, F_score, F1_avg = trainer.train()
        aji_list += [aji]
        mpq_list += [mpq]
        f1_det_list += [F1_det]
        f1_cls_list += [F1_avg]
        F_score = np.array(F_score).astype(np.float64)
        base_list = [i - 1 for i in args.base_list]
        f1_base_list += [np.mean(F_score[base_list])]
        novel_list = [i - 1 for i in args.novel_list]
        f1_novel_list += [np.mean(F_score[novel_list])]
        # multi_run_mean_f_score += F_score
        multi_run_mean_f_score += np.array(F_score).astype(np.float64)

    # print the final results
    logger.info('aji_mean_std: {}_{}'.format(np.mean(aji_list), np.std(aji_list)))
    logger.info('detection_f_score_mean_std: {}_{}'.format(np.mean(f1_det_list), np.std(f1_det_list)))
    logger.info('mpq_mean_std: {}_{}'.format(np.mean(mpq_list), np.std(mpq_list)))
    logger.info('classification_f_score_mean_std: {}_{}'.format(np.mean(f1_cls_list), np.std(f1_cls_list)))
    logger.info('classification_f_score_base_mean_std: {}_{}'.format(np.mean(f1_base_list), np.std(f1_base_list)))
    logger.info('classification_f_score_novel_mean_std: {}_{}'.format(np.mean(f1_novel_list), np.std(f1_novel_list)))

    multi_run_mean_f_score /= args.runs
    multi_run_mean_f_score = "F1_socre:" + "/".join("{}".format(f) for f in multi_run_mean_f_score)
    logger.info('mean_f1: {}'.format(multi_run_mean_f_score))



