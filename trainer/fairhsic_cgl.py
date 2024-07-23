"""
Original code:
    https://github.com/naver-ai/cgl_fairness
"""
from __future__ import print_function
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
        for epoch in range(self.epochs):
            self._train_epoch(train_loader, self.model, hsic=hsic, num_classes=num_classes)
            eval_loss, eval_acc, eval_deom, eval_deoa, eval_dp, eval_eoda = self.evaluate(self.model, test_loader, self.criterion)
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.0f} Test Acc: {:.2f} Test DEOM {:.2f} Test DEOA {:.2f}'.format
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
                res_path = f'{res_dir}/{self.args.version}_res.txt'
                with open(res_path, 'a') as file:
                    file.write('{:05.2f}  {:05.2f}  {:05.2f}  {:05.2f}  {:05.2f}  {:.0f}  {:.0f}  {:.1f}  {:.0f}'.format(eval_acc, eval_dp, eval_eoda, eval_deom, eval_deoa, self.args.batch_size, self.args.lamb, self.args.gamma, self.args.seed))  # 写入内容到文件
                    file.write('\n')
            if self.scheduler is not None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
        print('Training Finished!')

    def _train_epoch(self, train_loader, model, hsic=None, num_classes=2):
        model.train()
        running_acc = 0.0
        running_loss = 0.0
        fair_loss = 0.0
        num_classes = train_loader.dataset.num_classes
        idx_miss = train_loader.dataset.idx_miss
        # print("idx_miss的长度：")
        # print(len(idx_miss))
        # print("idx_miss:")
        # print(idx_miss)
        for i, data in enumerate(train_loader):
            ########### 每个batch都要计算fairness loss
            eval_acc = 0
            # Get the inputs
            inputs, _, groups, targets, indexes = data

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
            
            ############################### hsic_loss #########################
            f_s = outputs[-2]
            group_onehot = F.one_hot(groups).float()
            hsic_loss = 0
            ########### 仅使用敏感属性预测正确的进行计算fair_loss
            # 创建一个索引映射，将 indexes[0] 中的索引映射到 0 到 63 的范围
            index_mapping = {idx.item(): i for i, idx in enumerate(indexes[0])}
            # # 过滤 indexes[0]，只保留不在 idx_miss 中的索引，并映射到 0 到 63 范围
            valid_indexes = [index_mapping[i.item()] for i in indexes[0] if i.item() not in idx_miss]
            if len(valid_indexes) < 4:
                print("valid_index数量＜4，pass！")
                pass
            else:
                new_f_s = f_s[valid_indexes, :]
                new_group_onehot = group_onehot[valid_indexes, :]
            for l in range(num_classes):
                mask = targets == l
                mask_new = mask[valid_indexes]
                # hsic_loss += hsic.unbiased_estimator(f_s[mask], group_onehot[mask])
                hsic_loss += hsic.unbiased_estimator(new_f_s[mask_new], new_group_onehot[mask_new])
            
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

