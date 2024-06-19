import os
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
import pickle
from torch.utils.data import DataLoader

from models.QuadBranchNet.net_desc import QuadNet
from train_eval_utils import train_one_epoch, evaluate_one_epoch_patch_level
from dataset.dataset_monusac import Dataset_labeled, Base_Dataset, train_split_shot
from dataset.dataset_monusac import QuadCompose, QuadRandomRotate, QuadRandomFlip


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def create_model(device="cuda", weight_path='', num_type=5):
    model = QuadNet(pretrained_backbone='../Init_model/resnet50-0676ba61.pth',
                    num_types=num_type).to(device)

    if os.path.exists(weight_path):
        # 加载预训练权重
        # if "ImageNet" in weight_path:
        weight_dict = torch.load(weight_path, map_location=device)
        load_weight_dict = {k: v for k, v in weight_dict['model'].items()
                            if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weight_dict, strict=False)
        print('pretrained weights load complete!')
    return model


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device
    torch.cuda.set_device(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    if args.full_sup:
        save_dir = os.path.join(args.save_dir, args.dataset, 'full_sup')
    else:
        save_dir = os.path.join(args.save_dir, args.dataset, f'labeledShot_{args.shot}')

    os.makedirs(save_dir, exist_ok=True)

    import datetime
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    logger = get_logger(os.path.join(save_dir, f'train_log_{cur_time}.txt'))
    logger.info(args)

    assert args.runs == len(args.split_seed)

    ########################### Initialize the few-shot data loader ########################
    data_root = args.train_root
    labeled_transforms = QuadCompose([QuadRandomRotate(),
                                      QuadRandomFlip()])

    base_dataset = Base_Dataset(data_root=data_root)

    if not args.full_sup and not os.path.exists(f'../rand_split_file_{args.dataset}/labeled_shot_{args.shot}'):
        os.makedirs(f'../rand_split_file_{args.dataset}/labeled_shot_{args.shot}', exist_ok=True)
        train_split_shot(base_dataset=base_dataset, seed_list=args.split_seed,
                         shot=args.shot, dataset=args.dataset, num_type=args.num_type)

        # repeat the exp args.runs times
    num_run = args.runs if not args.full_sup else 1

    aji_list = []
    mpq_list = []
    f1_det_list = []
    f1_cls_list = []
    multi_run_mean_f_score = np.zeros(shape=(args.num_type - 1))

    # epochs = args.epochs * int(10 / args.shot)
    epochs = args.epochs
    print(f'training epochs {epochs}')

    for run in range(1, num_run + 1):
    # for run in [1, 3, 4, 5]:
        logger.info(f'#####################################run_{run}##########################################')
        os.makedirs(os.path.join(save_dir, f'run_{run}'),
                    exist_ok=True)

        if not args.full_sup:
            with open(f'../rand_split_file_{args.dataset}/labeled_shot_{args.shot}/split_{run}.pkl', 'rb') as f:
                split = pickle.load(f)
            labeled_idx = split['labeled']
        else:
            labeled_idx = np.arange(len(base_dataset))

        dataset_train = Dataset_labeled(data_root=args.train_root,
                                        idx=labeled_idx,
                                        transforms=labeled_transforms,
                                        mag=args.mag)

        dataset_test = Dataset_labeled(data_root=args.test_root,
                                       idx=[],
                                       mag=args.mag)

        logger.info(f'{len(dataset_train)} data for training, '
                    f' {len(dataset_test)} data for testing')

        loader_train = DataLoader(dataset=dataset_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=True)

        loader_test = DataLoader(dataset=dataset_test,
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 drop_last=False)

        ##################################### load pretrained model ############################
        # assign model
        model = create_model(args.device, num_type=args.num_type,
                             weight_path="/home/data1/my/Project/SGFSL/FullSup/exp_bk_0828/lizard/full_sup/run_1/model_20_aji_0.5972851468688212_mpq_0.44502307881542863.pth")

        # Init optimizer
        optimizer = torch.optim.Adam(
            [{'params': model.backbone.parameters()},
             {'params': model.conv_bot.parameters()},
             {'params': model.decoder.parameters()}],
            lr=args.lr, weight_decay=args.weight_decay)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
        #                                                  milestones=args.milestones,
        #                                                  gamma=0.1)

        for epoch in range(1, epochs + 1):
            accu_loss = train_one_epoch(model=model,
                                        optimizer=optimizer,
                                        loader_labeled=loader_train,
                                        device=args.device,
                                        num_type=args.num_type)

            logger.info(
                "EPOCH{}  loss:{:.3f}  bce:{:.3f}  dice:{:.3f} con_bce:{:.3f}  con_dice:{:.3f} "
                "cen_mse:{:.3f} type_bce:{:.3f} type_dice:{:.3f}" \
                "lr:{:.6f}".format(
                    epoch, accu_loss["total"] / len(loader_train),
                           accu_loss["sup"]["bce"] / len(loader_train),
                           accu_loss["sup"]["dice"] / len(loader_train),
                           accu_loss["sup"]["con_bce"] / len(loader_train),
                           accu_loss["sup"]["con_dice"] / len(loader_train),
                           accu_loss["sup"]["cen_mse"] / len(loader_train),
                           accu_loss["sup"]["type_bce"] / len(loader_train),
                           accu_loss["sup"]["type_dice"] / len(loader_train),

                    optimizer.state_dict()['param_groups'][0]['lr']))
            # scheduler.step()

            if epoch % epochs == 0 or epoch % 5 == 0:
                aji, F1_det, mpq, F_score, F1_avg = evaluate_one_epoch_patch_level(
                    model,
                    test_loader=loader_test,
                    test_dataset=dataset_test,
                    num_type=args.num_type,
                    device=args.device,
                    dataset=args.dataset,
                    is_draw=False,
                    num_run=run,
                    save_dir=save_dir
                )
                logger.info('aji: {}, detection_f1 {}, pq {}, cls_f1: {}'.format(aji, F1_det, mpq, F1_avg))
                logger.info('F_score: {}'.format(F_score))

            if epoch % epochs == 0 or epoch % 5 == 0:
                save_files = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': scheduler.state_dict(),
                    'epoch': epoch}
                torch.save(save_files,
                           "{}/run_{}/model_{}_aji_{}_mpq_{}.pth".format(save_dir, run, epoch, aji, mpq))
                aji_list += [aji]
                mpq_list += [mpq]
                f1_det_list += [F1_det]
                f1_cls_list += [F1_avg]
                multi_run_mean_f_score += np.array(F_score).astype(np.float64)

    # print the final results
    aji_out = '{}_{}'.format(np.mean(aji_list), np.std(aji_list))
    logger.info('aji_mean_std: {}'.format(aji_out))
    f_det_out = '{}_{}'.format(np.mean(f1_det_list), np.std(f1_det_list))
    logger.info('detection_f_score_mean_std: {}'.format(f_det_out))
    mpq_out = '{}_{}'.format(np.mean(mpq_list), np.std(mpq_list))
    logger.info('mpq_mean_std: {}'.format(mpq_out))
    f_cls_out = '{}_{}'.format(np.mean(f1_cls_list), np.std(f1_cls_list))
    logger.info('classification_f_score_mean_std: {}'.format(f_cls_out))
    multi_run_mean_f_score /= num_run
    multi_run_mean_f_score = "F1_socre:" + "/".join("{}".format(f) for f in multi_run_mean_f_score)
    logger.info('mean_f1: {}'.format(multi_run_mean_f_score))


if __name__ == '__main__':
    import argparse
    import numpy as np

    np.random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--visible_device', type=str, default="0", help="gpu can be visual")
    parser.add_argument('--use_unlabeled', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='lizard', help="dataset: lizard | pannuke")
    parser.add_argument('--split_seed', type=list, default=[123, 233, 422, 666, 999])

    parser.add_argument('--resume', type=str, default="", help="checkpoint的路径")
    parser.add_argument('--runs', type=int, default=5, help="times to repeat the experiment")
    parser.add_argument('--epochs', type=int, default=100, help="训练多少epoch")

    parser.add_argument('--full_sup', type=bool, default=True, help="is fully supervise the model training")
    parser.add_argument('--shot', type=float, default=50, help="N-way K-shot")
    parser.add_argument('--num_type', type=int, default=7, help="nuclei type category including background")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # 8 for 5 shot 2 for 1 shot
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--mag', type=str, default='20x')
    parser.add_argument('--train_root', type=str,
                        # default='/home/data1/my/dataset/consep/extracted_mirror/train/256x256_128x128/',
                        # default='/home/data1/my/dataset/monusac/extracted_mirror/train/256x256_256x256/',
                        default='/home/data1/my/dataset/lizard/Train/',
                        help="data root for training data")
    parser.add_argument('--test_root', type=str,
                        # default='/home/data1/my/dataset/consep/extracted_mirror/valid/256x256_256x256/',
                        # default='/home/data1/my/dataset/monusac/extracted_mirror/test/256x256_256x256/',
                        default='/home/data1/my/dataset/lizard/Test/',
                        help="data root for testing data")

    parser.add_argument('--test_save_dir', type=str, default="", help="验证时输出mask的路径")
    # parser.add_argument('--milestones', type=list, default=[60], help="学习率更新的epoch")
    parser.add_argument('--log_dir', type=str, default="./train_log", help="tensorboard文件的保存路径")
    parser.add_argument('--save_dir', type=str, default='./exp', help="模型的保存路径")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    print(args)
    main(args)
