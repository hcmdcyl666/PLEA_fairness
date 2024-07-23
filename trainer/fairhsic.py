
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
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEOM {:.2f} Test DEOA {:.2f}'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deom, eval_deoa))
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

    # def _train_epoch(self, epoch, train_loader, model, hsic=None, num_classes=3):
    def _train_epoch(self, epoch, train_loader, model, hsic=None, num_classes=2, before_length = 0, weights = None, groups_check = None):
        model.train()
        running_loss = 0.0
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        weights_set = set(weights.values()) # 一共就5个weight值
        weights_list = list(weights_set) # 必须先转为list
        weights_set = torch.tensor(weights_list)
        weights_set = weights_set.cuda(device=self.device) 
        j = 0

        fair_loss = 0
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
            ############################ hsic_loss #########################
            f_s = outputs[-2]
            ###################### 每个batch进行MC采样 ######################
            sample_weights = deepcopy(weights_batch)
            # 归一化权重
            sample_weights /= torch.sum(sample_weights)
            indices = torch.multinomial(sample_weights, num_samples=len(idx), replacement=True)
            group_onehot = F.one_hot(groups).float()
            new_f_s = f_s[indices, :]
            new_group_onehot = group_onehot[indices, :]

            hsic_loss = 0
            for l in range(num_classes):
                mask = targets == l
                # res = hsic.unbiased_estimator(f_s[mask], group_onehot[mask])
                res = hsic.unbiased_estimator(new_f_s[mask], new_group_onehot[mask])
                hsic_loss += res

            loss = loss + self.lamb * hsic_loss
            running_loss += loss.item()
            fair_loss += hsic_loss*10000 # 不然数字太小了存入文件为0

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        
        # 写入loss到文件
        train_loss = (running_loss / (i+1))*100
        fair_loss = (fair_loss / (i+1))*100
        res_dir = os.path.join(f'{project_path}/results',str(self.args.dataset), self.args.method, str(self.args.sv))
        check_log_dir(res_dir)
        res_path = f'{res_dir}/{self.args.version}_train_loss.txt' # 第一列总loss 第二列fair_loss
        with open(res_path, 'a') as file:
            file.write('{:.0f}  {:.0f}  {:.0f}  {:.0f}  {:.1f}  {:.0f}'.format(train_loss, fair_loss, self.args.batch_size, self.args.lamb, self.args.gamma, self.args.seed)) # 写入内容到文件
            file.write('\n')
            
