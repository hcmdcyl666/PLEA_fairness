"""
Original code:
    https://github.com/naver-ai/cgl_fairness
"""
from __future__ import print_function
import os
import torch
import torch.nn as nn
import time
from util_files.save_dis import load_data
from utils import check_log_dir, get_accuracy, get_samples_by_indices
import trainer
from torch.utils.data import DataLoader
import torch.autograd as autograd

# 获取当前工作目录的绝对路径
current_working_directory = os.getcwd()
project_path = current_working_directory
import datetime
# 获取当前日期
current_date = datetime.date.today()

import torch
from torch.utils.data import DataLoader

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.eta = args.eta
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.reweighting_target_criterion = args.reweighting_target_criterion
        self.slmode = True if args.sv < 1 else False
        self.version = args.version
        self.args = args
        self.cuda = False
    
    def train(self, train_loader, dl_idxs, du_idxs, test_loader, epochs, dummy_loader=None, writer=None):
        model = self.model
        model.train()
        num_groups = train_loader.dataset.num_groups
        num_classes = train_loader.dataset.num_classes
        print("mlp模型结构：")
        print(model)
        # print("base model: self.teacher")
        # print(self.teacher) # 仅考虑acc训练得到的target classifier
        # 计算影响函数并获取权重
        # dl_loader, du_loader = get_samples_by_indices(train_loader, dl_idxs)
        '''获取epsilon'''
        epsilons_save_dir = os.path.join(f'{project_path}/epsilons',str(self.args.dataset), str(self.args.sv), str(self.args.seed))
        file_name = f'{epsilons_save_dir}/epsilons.json'
        epsilons = load_data(file_name)
        # weight_set = [torch.exp(-influence) for influence in influences]
        n1 = len(du_idxs)
        n2 = len(dl_idxs)
        n = n1+n2
        weight_set_du = [1/n + epsilon for epsilon in epsilons] # 1+ε
        weight_set_dl = [1/n for _ in range(n2)]
        weight_dict = {}
        # 将 dl_loader 和 du_loader 的权重保存到字典中
        for idx, weight in zip(dl_idxs, weight_set_dl):
            weight_dict[idx] = weight
        for idx, weight in zip(du_idxs, weight_set_du):
            weight_dict[idx] = weight
            
        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model, weight_dict)   
            eval_loss, eval_acc, eval_deom, eval_deoa, eval_dp, eval_eoda = self.evaluate(self.model, test_loader, self.criterion)
            print('[{}/{}] Method: {} '
                'Test Loss: {:.3f} Test Acc: {:.2f} Test DEOM {:.2f} Test DEOA {:.2f}'.format
                (epoch + 1, epochs, self.method,
                eval_loss, eval_acc, eval_deom, eval_deoa))
            if epoch == self.epochs-1:
                res_dir = os.path.join(f'{project_path}/results',str(self.args.dataset), self.args.method, str(self.args.sv))
                check_log_dir(res_dir)
                res_path = f'{res_dir}/{self.args.version}_fairIF_res.txt'
                with open(res_path, 'a') as file:
                    file.write('{:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}'.format(eval_acc, eval_deom, eval_deoa, eval_dp, eval_eoda, self.args.batch_size, self.args.lamb, self.args.gamma, self.args.seed))  # 写入内容到文件
                    file.write('\n')

            if self.scheduler is not None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()

        print('Training Finished!')


    def _train_epoch(self, epoch, train_loader, model, weight_set):
        model.train()
        running_acc = 0.0
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, indexes = data
            labels = targets
            labels = labels.long()
            weights = torch.tensor([weight_set[i.item()] for i in indexes[0]])
            # 假设 self.cuda 是一个布尔值，表示是否使用GPU
            device = torch.device("cuda" if self.cuda else "cpu")

            # 将张量移动到适当的设备上
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights = weights.to(device) 
            groups = groups.to(device)
            # 将模型移动到相同的设备上
            model = model.to(device)
            # 执行前向传播
            outputs = model(inputs)
            loss = torch.sum(weights * (nn.CrossEntropyLoss(reduction='none')(outputs, labels)))
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def criterion(self, model, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)
