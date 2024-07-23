from __future__ import print_function
from copy import deepcopy
import os

import numpy as np
import torch
import torch.nn.functional as F
import time
from util_files.save_dis import load_data
from utils import get_accuracy
import trainer
from .hsic import RbfHSIC
import torch.nn as nn
# 获取当前工作目录的绝对路径
current_working_directory = os.getcwd()
project_path = current_working_directory
import datetime
from utils import check_log_dir, make_log_name, set_seed
# 获取当前日期
current_date = datetime.date.today()

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lamb = args.lamb
        self.sigma = args.sigma
        self.kernel = args.kernel
        self.slmode = True if args.sv < 1 else False
        self.version = args.version
        self.args = args

    def train(self, train_loader, test_loader, epochs, writer=None):
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups
        hsic = RbfHSIC(1, 1)
        '''
        获取样本的weight
        '''
        weights_save_dir = os.path.join(f'{project_path}/weights', str(self.args.dataset), str(self.args.sv), str(self.args.seed))
        weights_path = f'{weights_save_dir}/weights_{self.args.dis_metric}_{self.args.w_method}_{self.args.gamma}.json' # 记录有每个样本的id：weight
        json_data = load_data(weights_path)
        weights = {int(k): v for k, v in json_data.items()}
        for epoch in range(self.epochs):
            self._train_epoch(epoch, train_loader, self.model, hsic=hsic, num_classes=num_classes, before_length=0, weights=weights, groups_check = None)
            eval_loss, eval_acc, eval_deom, eval_deoa, eval_dp, eval_eoda = self.evaluate(self.model, test_loader, self.criterion)
            # 写入loss到文件
            res_dir = os.path.join(f'{project_path}/results',str(self.args.dataset), self.args.method, str(self.args.sv))
            check_log_dir(res_dir)
            res_path = f'{res_dir}/{self.args.version}_test_loss.txt' # 测试集utility loss
            with open(res_path, 'a') as file:
                file.write('{:.0f}  {:.0f}  {:.0f}  {:.1f}  {:.0f}'.format(eval_loss, self.args.batch_size, self.args.lamb, self.args.gamma, self.args.seed)) # 写入内容到文件
                file.write('\n')
            if epoch == self.epochs-1:
                res_dir = os.path.join(f'{project_path}/results',str(self.args.dataset), self.args.method, str(self.args.sv))
                check_log_dir(res_dir)
                res_path = f'{res_dir}/{self.args.dis_metric}_{self.args.w_method}_{self.args.dis_method}_res.txt'
                with open(res_path, 'a') as file:
                    file.write('{:05.2f}  {:05.2f}  {:05.2f}  {:05.2f}  {:05.2f}  {:.0f}  {:.0f}  {:.1f}  {:.0f}'.format(eval_acc, eval_dp, eval_eoda, eval_deom, eval_deoa, self.args.batch_size, self.args.lamb, self.args.gamma, self.args.seed))  # 写入内容到文件
                    file.write('\n')

            if self.scheduler is not None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()

        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model, hsic=None, num_classes=2, before_length = 0, weights = None, groups_check = None):
        model.train()
        running_acc = 0.0
        running_loss = 0.0
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        weights_set = set(weights.values()) 
        weights_list = list(weights_set) # 必须先转为list
        weights_set = torch.tensor(weights_list)
        weights_set = weights_set.cuda(device=self.device) 
        j = 0
        for i, data in enumerate(train_loader):
            ########### 每个batch都要计算fairness loss ###########
            eval_acc = 0
            inputs, _, groups, targets, (idx, _) = data
            weights_batch = []
            for key in idx.tolist():
                weights_batch.append(weights[key])
            weights_batch = torch.tensor(weights_batch)
            weights_batch = weights_batch.cuda(device=self.device)

            labels = targets
            if self.cuda:
                inputs = inputs.cuda(self.device)
                labels = labels.cuda(self.device)
                groups = groups.long().cuda(self.device)

            outputs = model(inputs, get_inter=True) # 输出中间embedding层
            stu_logits = outputs[-1] # 输出是每一行都有俩概率值，但是不一定和为1（还没有进行归一化）
            loss = self.criterion(stu_logits, labels)
            eval_acc = get_accuracy(stu_logits, labels)
            running_acc += eval_acc
            ############################ hsic_loss #########################
            ############### 根据D_U中样本敏感属性的预测区间分bin来分别计算HSIC ##################
            f_s = outputs[-2]
            group_onehot = F.one_hot(groups).float()
            hsic_loss = 0
            for l in range(num_classes):
                mask = targets == l
                bin_sum = 0
                for w in weights_set:
                    w_mask = weights_batch[mask] == w
                    s = sum(w_mask) # s是当前batch中属于当前bin的样本数量
                    if s != 0 and s != 1:
                        bin_sum += w*s.float()
                for w in weights_set:
                    res = 0
                    w_mask = weights_batch[mask] == w
                    s = sum(w_mask)
                    if s != 0 and s != 1: # 这种情况就没有求hsic的必要了
                        if s >= 4:
                            res = hsic.unbiased_estimator(f_s[mask][w_mask], group_onehot[mask][w_mask])
                        else:
                            res = hsic.biased_estimator(f_s[mask][w_mask], group_onehot[mask][w_mask]) # 这里必须用biased方法 因为N可能小于4
                       
                        hsic_loss += (w*s/bin_sum)*res

            loss = loss + self.lamb * hsic_loss
            running_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

