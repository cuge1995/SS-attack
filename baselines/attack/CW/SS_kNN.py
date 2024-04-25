"""Implementation of optimization based attack,
    CW Attack for ROBUST point perturbation.
Based on AAAI'20: Robust Adversarial Objects against Deep Learning Models.
"""

import pdb
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=0.5, scale_high=1.5, translate_range=0.0, no_z_aug=False):
        """
        :param scale_low:
        :param scale_high:
        :param translate_range:
        :param no_z: no translation and scaling along the z axis
        """
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range
        self.no_z_aug = no_z_aug

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            # xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            # if self.no_z_aug:
            #     xyz1[2] = 1.0
            #     xyz2[2] = 0.0
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) #+ torch.from_numpy(xyz2).float().cuda()

        return pc

def shear(pointcloud,severity=1):
    pointcloud = pointcloud.transpose(1, 2)
    B, N, C = pointcloud.shape
    # c = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
    cl = 0.05
    ch = 0.10
    a = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])
    b = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])
    d = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])
    e = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])
    f = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])
    g = np.random.uniform(cl-0.05,ch+0.05) * np.random.choice([-1,1])

    matrix = torch.from_numpy(np.array([[1,0,b],[d,1,e],[f,0,1]])).view(1,3,3).expand(B,-1,-1).cuda().float()
    new_pc = torch.matmul(pointcloud,matrix) #.astype('float32')
    new_pc = new_pc.transpose(2, 1)
    return new_pc



class SSCWKNN:
    """Class for CW attack.
    """

    def __init__(self, model, adv_func, dist_func, clip_func,
                 attack_lr=1e-3, num_iter=2500):
        """CW attack by kNN attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            dist_func (function): distance metric
            clip_func (function): clipping function
            attack_lr (float, optional): lr for optimization. Defaults to 1e-3.
            num_iter (int, optional): max iter num in every search step. Defaults to 2500.
        """

        self.model = model.cuda()
        self.model.eval()

        self.adv_func = adv_func
        self.dist_func = dist_func
        self.clip_func = clip_func
        self.attack_lr = attack_lr
        self.num_iter = num_iter

    def attack(self, data, target):
        """Attack on given data to target.

        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.LongTensor): target output, [B]
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        data = data.transpose(1, 2).contiguous()
        ori_data = data.clone().detach()
        ori_data.requires_grad = False

        # points and normals
        if ori_data.shape[1] == 3:
            normal = None
        else:
            normal = ori_data[:, 3:, :]
            ori_data = ori_data[:, :3, :]
        target = target.long().cuda().detach()

        # init variables with small perturbation
        adv_data = ori_data.clone().detach() + \
            torch.randn((B, 3, K)).cuda() * 1e-7
        adv_data.requires_grad_()
        opt = optim.Adam([adv_data], lr=self.attack_lr, weight_decay=0.)

        adv_loss = torch.tensor(0.).cuda()
        dist_loss = torch.tensor(0.).cuda()

        total_time = 0.
        forward_time = 0.
        backward_time = 0.
        clip_time = 0.

        # there is no binary search in this attack
        # just longer iterations of optimization
        for iteration in range(self.num_iter):
            t1 = time.time()

            # forward passing
            r = np.random.rand(1)
            if r <= 0.7:
                ## shear
                r1 = np.random.rand(1)
                if r1 <= 0.7:
                    nadv_data = shear(adv_data)
                    # nadv_data = cut_points_knn(adv_data)
                    # nadv_data = adv_data + torch.randn((B, 3, K)).cuda() * 1e-4
                    logits = self.model(nadv_data)
                else:
                    scale = PointcloudScaleAndTranslate()
                    nadv_data = scale(adv_data.data)
                    logits = self.model(nadv_data)
            else:
                logits = self.model(adv_data)
            # logits = self.model(adv_data)  # [B, num_classes]
            if isinstance(logits, tuple):  # PointNet
                logits = logits[0]

            t2 = time.time()
            forward_time += t2 - t1

            # print
            pred = torch.argmax(logits, dim=1)  # [B]
            success_num = (pred == target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('Iteration {}/{}, success {}/{}\n'
                      'adv_loss: {:.4f}, dist_loss: {:.4f}'.
                      format(iteration, self.num_iter, success_num, B,
                             adv_loss.item(), dist_loss.item()))

            # compute loss and backward
            adv_loss = self.adv_func(logits, target).mean()

            # in the official tensorflow code, they use sum instead of mean
            # so we multiply num_points as sum
            dist_loss = self.dist_func(
                adv_data.transpose(1, 2).contiguous(),
                ori_data.transpose(1, 2).contiguous()).mean() * K

            loss = adv_loss + dist_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            t3 = time.time()
            backward_time += t3 - t2

            # clipping and projection!
            adv_data.data = self.clip_func(adv_data.clone().detach(),
                                           ori_data, normal)

            t4 = time.time()
            clip_time = t4 - t3
            total_time += t4 - t1

            if iteration % 100 == 0:
                print('total time: {:.2f}, for: {:.2f}, '
                      'back: {:.2f}, clip: {:.2f}'.
                      format(total_time, forward_time,
                             backward_time, clip_time))
                total_time = 0.
                forward_time = 0.
                backward_time = 0.
                clip_time = 0.
                torch.cuda.empty_cache()

        # end of CW attack
        with torch.no_grad():
            logits = self.model(adv_data)  # [B, num_classes]
            if isinstance(logits, tuple):  # PointNet
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)  # [B]
            success_num = (pred == target).\
                sum().detach().cpu().item()

        # return final results
        print('Successfully attack {}/{}'.format(success_num, B))

        # in their implementation, they estimate the normal of adv_pc
        # we don't do so here because it's useless in our task
        adv_data = adv_data.transpose(1, 2).contiguous()  # [B, K, 3]
        adv_data = adv_data.detach().cpu().numpy()  # [B, K, 3]
        return adv_data, success_num
