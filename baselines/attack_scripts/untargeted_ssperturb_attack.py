"""Targeted kNN attack."""

import os
from tqdm import tqdm
import argparse
import numpy as np
import psutil

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import sys
sys.path.append('../')
sys.path.append('./')

from config import BEST_WEIGHTS
from config import MAX_KNN_BATCH as BATCH_SIZE
from dataset import ScanObjectNNDataLoader, ModelNet40Attack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from util.utils import str2bool, set_seed
from attack import SSCWPerturb
from attack import CrossEntropyAdvLoss, UntargetedLogitsAdvLoss
from attack import ChamferkNNDist
from attack import ProjectInnerClipLinf
from attack import L2Dist
from attack import ClipPointsLinf




def get_model(cfg):
    if args.model.lower() == 'simpleview':
        model = models.MVModel(
            task="cls_trans",
            dataset="scanobjectnn")
    elif args.model.lower() == 'rscnn':
        model = models.RSCNN(
            task="cls",
            dataset="scanobjectnn")
    elif args.model.lower() == 'pointnet2':
        model = models.PointNet2(
            task="cls",
            dataset="scanobjectnn")
    elif args.model.lower() == 'dgcnn':
        model = models.DGCNN(
            task="cls",
            dataset="scanobjectnn")
    elif args.model.lower() == 'pointnet':
        model = models.PointNet(
            task="cls_trans",
            dataset="scanobjectnn")
    elif args.model.lower() == 'pct':
        model = models.Pct(
            task="cls",
            dataset="scanobjectnn")
    elif args.model.lower() == 'pointMLP':
        model = models.pointMLP(
            task="cls",
            dataset="scanobjectnn")
    elif args.model.lower() == 'pointMLP2':
        model = models.pointMLP2(
            task="cls",
            dataset="scanobjectnn")
    elif args.model.lower() == 'curvenet':
        model = models.CurveNet(
            task="cls",
            dataset="scanobjectnn")
    elif args.model.lower() == 'gdanet':
        model = models.GDANET(
            task="cls",
            dataset="scanobjectnn")
    else:
        assert False

    return model



def attack():
    model.eval()
    all_ori_pc = []
    all_adv_pc = []
    all_real_lbl = []
    num = 0
    for pc, label in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True)

        # attack!
        _, best_pc, success_num = attacker.attack(pc, label)

        # results
        num += success_num
        all_ori_pc.append(pc.detach().cpu().numpy())
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())

    # accumulate results
    all_ori_pc = np.concatenate(all_ori_pc, axis=0)  # [num_data, K, 3]
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    return all_ori_pc, all_adv_pc, all_real_lbl, num

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='official_data/modelnet40_normal_resampled')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='scanobjectnn', metavar='N',
                        choices=['scanobjectnn', 'remesh_mn40', 'opt_mn40', 'conv_opt_mn40', 'aug_mn40'])
    parser.add_argument('--batch_size', type=int, default=-1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=15.,
                        help='min margin in logits adv loss')
    parser.add_argument('--budget', type=float, default=0.08,
                        help='clip budget')
    parser.add_argument('--attack_lr', type=float, default=1e-3,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step', type=int, default=10, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=500, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_points]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_points]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)


    # dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    cudnn.benchmark = True

    # build model
    import sys
    sys.path.append('/home/adlab2/project/ModelNet40-C')
    import models

    model = get_model(args)



    # if args.model.lower() == 'dgcnn':
    #     model = DGCNN(args.emb_dims, args.k, output_channels=40)
    # elif args.model.lower() == 'pointnet':
    #     model = PointNetCls(k=40, feature_transform=args.feature_transform)
    # elif args.model.lower() == 'pointnet2':
    #     model = PointNet2ClsSsg(num_classes=40)
    # elif args.model.lower() == 'pointconv':
    #     model = PointConvDensityClsSsg(num_classes=40)
    # else:
    #     print('Model not recognized')
    #     exit(-1)

    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))
    try:
        model.load_state_dict(state_dict['model_state'])
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict['model_state'].items()}
        model.load_state_dict(state_dict)
    
    model = model.cuda()

    # model = DistributedDataParallel(
    #     model.cuda(), device_ids=[args.local_rank])
    # model = torch.nn.DataParallel(model).cuda()

    # setup attack settings
    # setup attack settings
    if args.adv_func == 'logits':
        adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()
    dist_func = L2Dist()
    clip_func = ClipPointsLinf(budget=args.budget)
    # hyper-parameters from their official tensorflow code
    attacker = SSCWPerturb(model, adv_func, dist_func,
                         attack_lr=args.attack_lr,
                         init_weight=10., max_weight=80.,
                         binary_step=args.binary_step,
                         num_iter=args.num_iter, clip_func=clip_func)

    # attack
    test_set = ScanObjectNNDataLoader(split='test')
    # test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, 
                    shuffle=False, num_workers=10, 
                    drop_last=False)

    # run attack
    ori_data, attacked_data, real_label, success_num = attack()

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)

    # save results
    save_path = './attack/results/{}_{}/Perturb'.\
        format(args.dataset, args.num_points)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.adv_func == 'logits':
        args.adv_func = 'logits_kappa={}'.format(args.kappa)
    save_name = 'UkNN-{}-{}-success_{:.4f}-rank_{}.npz'.\
        format(args.model, args.budget,
               success_rate, args.local_rank)
    np.savez(os.path.join(save_path, save_name),
             ori_pc=ori_data.astype(np.float32),
             test_pc=attacked_data.astype(np.float32),
             test_label=real_label.astype(np.uint8))