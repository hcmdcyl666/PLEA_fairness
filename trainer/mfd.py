"""
Original code:
    https://github.com/sangwon79/Fair-Feature-Distillation-for-Visual-Recognition
"""
from __future__ import print_function
from copy import deepcopy
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from utils import get_accuracy
from util_files.save_dis import load_data
import trainer
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

        # print("self.teacher")
        # print(self.teacher) # 仅考虑acc训练得到的target classifier

        '''
        获取样本的weight
        '''
        weights_save_dir = os.path.join(f'{project_path}/weights', str(self.args.dataset), str(self.args.sv), str(self.args.seed))
        weights_path = f'{weights_save_dir}/weights_{self.args.dis_metric}_{self.args.w_method}_{self.args.gamma}.json' # 记录有每个样本的id：weight
        json_data = load_data(weights_path)
        weights = {int(k): v for k, v in json_data.items()}

        distiller = MMDLoss(w_m=self.lamb, sigma=self.sigma,
                            num_classes=num_classes, num_groups=num_groups, kernel=self.kernel)
        for epoch in range(self.epochs):
            self._train_epoch(epoch, train_loader, self.model, self.teacher, distiller=distiller, weights=weights)
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

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller=None, weights=None):
        model.train()
        teacher.eval()

        running_acc = 0.0
        running_loss = 0.0

        weights_set = set(weights.values()) # 一共就5个weight值
        weights_list = list(weights_set) # 必须先转为list
        weights_set = torch.tensor(weights_list)
        weights_set = weights_set.cuda(device=self.device) 

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, (idx, _) = data
            labels = targets

            weights_batch = []
            for key in idx.tolist():
                weights_batch.append(weights[key])
            weights_batch = torch.tensor(weights_batch)
            weights_batch = weights_batch.cuda(device=self.device)

            if self.cuda:
                inputs = inputs.cuda(self.device)
                labels = labels.cuda(self.device)
                groups = groups.long().cuda(self.device)

            # 把teacher挪到设定的cuda设备上
            teacher = teacher.to(self.device)
            # t_inputs = inputs.to(self.device)
            t_outputs = teacher(inputs, get_inter=True)
            
            outputs = model(inputs, get_inter=True)
            stu_logits = outputs[-1]
            loss = self.criterion(stu_logits, labels)
            running_acc += get_accuracy(stu_logits, labels)

            f_s = outputs[-2]
            f_t = t_outputs[-2].detach() # teacher模型和student模型学习到的特征表示
            f_t.to(self.device)
            ###################### 每个batch进行MC采样 ######################
            sample_weights = deepcopy(weights_batch)
            # 归一化权重
            sample_weights /= torch.sum(sample_weights)
            indices = torch.multinomial(sample_weights, num_samples=len(idx), replacement=True)
            # # 将 indices 移动到 t_device 设备上
            # indices = indices.to(self.device)
            new_groups = groups[indices]
            new_labels = labels[indices]
            new_f_s = f_s[indices, :]
            new_f_t = f_t[indices, :]

            # mmd_loss = distiller.forward(f_s, f_t, groups=groups, labels=labels)
            mmd_loss = distiller.forward(new_f_s, new_f_t, groups=new_groups, labels=new_labels)

            loss = loss + mmd_loss
            running_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class MMDLoss(nn.Module):
    def __init__(self, w_m, sigma, num_groups, num_classes, kernel):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1)

        mmd_loss = 0
        with torch.no_grad():
            _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

        for c in range(self.num_classes):
            if len(teacher[labels == c]) == 0:
                continue
            for g in range(self.num_groups):
                if len(student[(labels == c) * (groups == g)]) == 0:
                    continue
                K_TS, _ = self.pdist(teacher[labels == c], student[(labels == c) * (groups == g)],
                                     sigma_base=self.sigma, sigma_avg=sigma_avg,  kernel=self.kernel)
                K_SS, _ = self.pdist(student[(labels == c) * (groups == g)], student[(labels == c) * (groups == g)],
                                     sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                K_TT, _ = self.pdist(teacher[labels == c], teacher[labels == c], sigma_base=self.sigma,
                                     sigma_avg=sigma_avg, kernel=self.kernel)

                mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()
        
        # print("mmd_loss:")
        # print(mmd_loss)
        loss = (1/2) * self.w_m * mmd_loss
        return loss
    @staticmethod
    # 分布p之间的距离dist
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()
                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base)*sigma_avg))
                # res是高斯核矩阵(n*n)
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg
