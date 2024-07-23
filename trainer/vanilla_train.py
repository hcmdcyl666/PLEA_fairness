"""
Original code:
    https://github.com/naver-ai/cgl_fairness
"""
from __future__ import print_function
import os
import time
import torch
from utils import get_accuracy
import trainer
# 获取当前工作目录的绝对路径
current_working_directory = os.getcwd()
project_path = current_working_directory
import datetime
# 获取当前日期
current_date = datetime.date.today()
from torch import nn
from util_files.save_to_file import *
from utils import check_log_dir
class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        if criterion == None:
            criterion = self.criterion
        model = self.model
        print("mlp模型结构：")
        print(model)
        model.train()
        print("criterion:")
        print(criterion)

        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model, criterion)
            eval_loss, eval_acc, eval_deom, eval_deoa, eval_dp, eval_eoda = self.evaluate(self.model, test_loader, criterion)
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
                res_path = f'{res_dir}/vanilla_res.txt'
                with open(res_path, 'a') as file:
                    file.write('{:05.2f}  {:05.2f}  {:05.2f}  {:05.2f}  {:05.2f}  {:.0f}  {:.0f}  {:.1f}  {:.0f}'.format(eval_acc, eval_dp, eval_eoda, eval_deom, eval_deoa, self.args.batch_size, self.args.lamb, self.args.gamma, self.args.seed))  # 写入内容到文件
                    file.write('\n')

            if self.scheduler is not None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model, criterion=None, weights = None):
        model.train()
        running_acc = 0.0
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
            outputs = model(inputs)
            ############# 训练Lbg(x)
            # 添加高斯噪声
            # noise = torch.normal(0, 0.1, size=inputs.shape)
            # noise = noise.cuda(device=self.device)
            # noisy_inputs = inputs + noise
            # outputs = model(noisy_inputs)

            if criterion is not None:
                loss = criterion(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            