import glob
import os
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import logging
from torch.utils.data import DataLoader
from models.QuadBranchNet.net_desc import QuadNet
from train_eval_utils import evaluate_one_epoch_patch_level
from dataset.dataset_monusac import Dataset_labeled
from utils.logger import get_logger

def create_model(device="cuda", weight_path='', num_type=5):
    model = QuadNet(pretrained_backbone='../Init_model/resnet50-0676ba61.pth',
                        num_types=num_type).to(device)
    if os.path.exists(weight_path):
        # 加载预训练权重
        weight_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(weight_dict)
    return model


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device
    torch.cuda.set_device(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    for shot in [50]:
        print(f'shot {shot}')
        save_dir = os.path.join(args.save_dir, 'ext_' + args.ext_dataset + '_tsk_' + args.tsk_dataset,
                                f'labeledShot_{shot}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        import datetime
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        logger = get_logger(os.path.join(save_dir, f'test_patch_lv_log_{cur_time}.txt'))
        logger.info(args)

        assert args.runs == len(args.split_seed)
        ########################## Generate train & test data list for each dataset #############
        num_run = args.runs
        f1_det_list = []
        f1_cls_list = []
        aji_list = []
        mpq_list = []
        f1_base_list = []
        f1_novel_list = []
        multi_run_mean_f_score = np.zeros(shape=(args.num_type - 1))

        dataset_test = Dataset_labeled(data_root=args.test_root,
                                       idx=[],
                                       name=args.tsk_dataset)

        loader_test = DataLoader(dataset=dataset_test,
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 drop_last=False)

        # for run in range(1, num_run + 1):
        for run in range(1, 6): # draw results for the first run
            logger.info(f'#####################################run_{run}##########################################')
            model = create_model(args.device, weight_path='', num_type=args.num_type)
            # weight_path = ["/home/data1/my/Project/SGFSL/FullSup/exp/monusac/labeledShot_30/run_1/model_25_aji_0.5256120960145282_mpq_0.328024064711192.pth"]
            weight_path = glob.glob(os.path.join(save_dir, f'run_{run}', '*.pth'))
            assert len(weight_path) == 1
            weight_dict = torch.load(weight_path[0], map_location="cuda")
            model.load_state_dict(weight_dict['model'], strict=True)

            aji, F1_det, mpq, F_score, F1_avg = evaluate_one_epoch_patch_level(
                                                               model,
                                                               test_loader=loader_test,
                                                               test_dataset=dataset_test,
                                                               num_type=args.num_type,
                                                               device=args.device,
                                                               dataset=args.tsk_dataset,
                                                               is_draw=False,
                                                               num_run=run,
                                                               save_dir=save_dir)


            logger.info('aji: {}, detection_f1 {}, pq {}, cls_f1: {}'.format(aji, F1_det,
                                                                             mpq, F1_avg))
            logger.info('F_score: {}'.format(F_score))

            aji_list += [aji]
            mpq_list += [mpq]
            f1_det_list += [F1_det]
            f1_cls_list += [F1_avg]
            F_score = np.array(F_score).astype(np.float64)
            base_list = [i - 1 for i in args.base_list]
            f1_base_list += [np.mean(F_score[base_list])]
            novel_list = [i - 1 for i in args.novel_list]
            f1_novel_list += [np.mean(F_score[novel_list])]
            multi_run_mean_f_score += F_score

            # inst_count = 0
            # base_count = 0
            # novel_count =0
            # inst_type_count = dict(inst_type_count)
            # if 1 in inst_type_count:
            #     inst_type_count.pop(1)
            # for key in inst_type_count:
            #     if key in args.base_list:
            #         base_count += inst_type_count[key]
            #     if key in inst_type_count

        logger.info('aji_mean_std: {}_{}'.format(np.mean(aji_list), np.std(aji_list)))
        logger.info('detection_f_score_mean_std: {}_{}'.format(np.mean(f1_det_list), np.std(f1_det_list)))
        logger.info('mpq_mean_std: {}_{}'.format(np.mean(mpq_list), np.std(mpq_list)))
        logger.info('classification_f_score_mean_std: {}_{}'.format(np.mean(f1_cls_list), np.std(f1_cls_list)))
        logger.info('classification_f_score_base_mean_std: {}_{}'.format(np.mean(f1_base_list), np.std(f1_base_list)))
        logger.info('classification_f_score_novel_mean_std: {}_{}'.format(np.mean(f1_novel_list), np.std(f1_novel_list)))

        multi_run_mean_f_score /= num_run
        multi_run_mean_f_score = "F1_socre:" + "/".join("{}".format(f) for f in multi_run_mean_f_score)
        logger.info('aji_mean_f1: {}'.format(multi_run_mean_f_score))


if __name__ == '__main__':
    import argparse
    import numpy as np
    np.random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--visible_device', type=str, default="0", help="gpu can be visual")
    parser.add_argument('--amp', type=bool, default=True, help="是否使用混合精度")
    parser.add_argument('--use_unlabeled', type=bool, default=False)
    parser.add_argument('--ext_dataset', type=str, default='consep', help="dataset: lizard | pannuke | consep | monusac")
    parser.add_argument('--tsk_dataset', type=str, default='monusac', help="dataset: lizard | pannuke | consep | monusac")
    parser.add_argument('--split_seed', type=list, default=[123, 233, 422, 666, 999])
    parser.add_argument('--weight_path', type=str,
                        default="",
                        help="训练权重")
    # parser.add_argument('--resume', type=str, default="", help="checkpoint的路径")
    parser.add_argument('--runs', type=int, default=5, help="times to repeat the experiment")

    # parser.add_argument('--shot', type=float, default=1, help="有标签与所有数据的比例")
    parser.add_argument('--num_type', type=int, default=5, help="nuclei type category including background")
    # 8 for 5 shot 2 for 1 shot
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--mask_shape', type=int, default=256)

    # parser.add_argument('--novel_list', type=list, default=[1, 4], help="")
    # parser.add_argument('--base_list', type=list, default=[2, 3], help="")
    parser.add_argument('--novel_list', type=list, default=[3, 4], help="")
    parser.add_argument('--base_list', type=list, default=[1, 2], help="")

    # parser.add_argument('--novel_list', type=list, default=[4, 5, 6], help="")
    # parser.add_argument('--base_list', type=list, default=[1, 2, 3], help="")

    parser.add_argument('--test_root', type=str,
                        default='/home/data1/my/dataset/monusac/extracted_mirror/test/256x256_256x256/',
                        # default='/home/data1/my/dataset/consep/extracted_mirror/valid/256x256_256x256/',
                        # default='/home/data1/my/dataset/lizard/Test/',
                        help="data root for testing data")
    parser.add_argument('--log_dir', type=str, default="./train_log", help="tensorboard文件的保存路径")
    parser.add_argument('--save_dir', type=str, default='./exp', help="模型的保存路径")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    print(args)
    main(args)
