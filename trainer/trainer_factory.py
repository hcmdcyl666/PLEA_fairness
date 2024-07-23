"""
Original code:
    https://github.com/naver-ai/cgl_fairness
"""
import torch
import numpy as np
import os
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
from sklearn.metrics import confusion_matrix
from utils import make_log_name


class TrainerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(method, **kwargs):
        if method == 'scratch':
            import trainer.vanilla_train as trainer
        elif method == 'mfd':
            import trainer.mfd as trainer
        elif method == 'mfd_cgl':
            import trainer.mfd_cgl as trainer
        elif method == 'mfd_bin':
            import trainer.mfd_bin as trainer
        elif method == 'fairhsic':
            import trainer.mfd_bin as trainer
        elif method == 'fairif':
            import trainer.fair_if as trainer
        elif method == 'fairhsic_bin':
            import trainer.fairhsic_bin as trainer
        elif method == 'fairhsic_cgl':
            import trainer.fairhsic_cgl as trainer
        elif method == 'adv':
            import trainer.adv_debiasing as trainer
        elif method == 'reweighting':
            import trainer.reweighting as trainer
        elif method == 'groupdro':
            import trainer.groupdro as trainer
        elif method == 'groupdro_ori':
            import trainer.groupdro as trainer
        else:
            raise Exception('Not allowed method')
        return trainer.Trainer(**kwargs)


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this.
    '''
    def __init__(self, model, args, optimizer, teacher=None):
        self.get_inter = args.get_inter

        self.record = args.record
        self.cuda = args.cuda
        self.device = args.device
        self.t_device = args.t_device
        self.term = args.term
        self.lr = args.lr
        self.epochs = args.epochs
        self.method = args.method
        self.model_name =args.model
        self.model = model
        self.teacher = teacher
        self.optimizer = optimizer
        self.optim_type = args.optimizer
        self.log_dir = args.log_dir
        self.criterion=torch.nn.CrossEntropyLoss()
        self.scheduler = None

        self.log_name = make_log_name(args)
        self.log_dir = os.path.join(args.log_dir, args.date, args.dataset, args.method)
        self.save_dir = os.path.join(args.save_dir, args.date, args.dataset, args.method)

        if self.optim_type == 'Adam' and self.optimizer is not None:
            self.scheduler = ReduceLROnPlateau(self.optimizer)
        elif self.optim_type == 'AdamP' and self.optimizer is not None:
            if self.epochs < 100:
                t_max = self.epochs
            elif self.epochs == 200:
                t_max = 66
            self.scheduler = CosineAnnealingLR(self.optimizer, t_max)
        else:
            self.scheduler = MultiStepLR(self.optimizer, [60, 120, 180], gamma=0.1)
            #self.scheduler = MultiStepLR(self.optimizer, [30, 60, 90], gamma=0.1)

    def scale_and_return(self, eval_loss, eval_acc, eval_max_eopp, eval_avg_eopp, dp_positive, eval_avg_eod):
        # 将每个值乘以100
        eval_loss *= 10000
        eval_acc *= 100
        eval_max_eopp *= 100
        eval_avg_eopp *= 100
        dp_positive *= 100
        eval_avg_eod *= 100
    
        return eval_loss, eval_acc, eval_max_eopp, eval_avg_eopp, dp_positive, eval_avg_eod

    def evaluate(self, model, loader, criterion, device=None):
        model.eval()
        num_groups = loader.dataset.num_groups
        num_classes = loader.dataset.num_classes
        device = self.device if device is None else device

        eval_acc = 0
        eval_loss = 0
        eval_eopp_list = torch.zeros(num_groups, num_classes).cuda(device)
        eval_dp_list = torch.zeros(num_groups, num_classes).cuda(device)
        eval_eod_list = torch.zeros(num_groups, 2).cuda(device) # 有且仅有2列
        eval_data_count = torch.zeros(num_groups, num_classes).cuda(device)

        if 'Custom' in type(loader).__name__:
            loader = loader.generate()
        with torch.no_grad():
            for j, eval_data in enumerate(loader):
                if j == 100:
                    break
                # Get the inputs
                inputs, _, groups, classes, _ = eval_data
                #
                labels = classes
                if self.cuda:
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    groups = groups.cuda(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item() * len(labels)
                preds = torch.argmax(outputs, 1)
                acc = (preds == labels).float().squeeze()
                eval_acc += acc.sum()

                for g in range(num_groups):
                    for l in range(num_classes):
                        group_mask = (groups == g)
                        label_mask = (labels == l)
                        pred_label_mask = (preds == l)
                        group_label_mask = group_mask & label_mask
                        group_pred_label_mask = group_mask & pred_label_mask

                        eval_eopp_list[g, l] += acc[group_label_mask].sum()
                        eval_dp_list[g, l] += group_pred_label_mask.float().sum()
                        eval_data_count[g, l] += group_label_mask.float().sum()

                # 计算 TPR 和 FPR（EOD）
                for g in range(num_groups):
                    tp = acc[(groups == g) & (labels == 1) & (preds == 1)].sum()
                    # 假设 labels, preds, 和 groups 都是张量
                    fn = ((labels == 1) & (preds == 0) & (groups == g)).float().sum()
                    fp = ((labels == 0) & (preds == 1) & (groups == g)).float().sum()
                    tn = acc[(groups == g) & (labels == 0) & (preds == 0)].sum()
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else torch.tensor(0.0)
                    
                    eval_eod_list[g, 0] += tpr
                    eval_eod_list[g, 1] += fpr

            ##################### 最终计算 ####################
            # 计算和输出最终的指标
            eval_loss = eval_loss / eval_data_count.sum()
            eval_acc = eval_acc / eval_data_count.sum()
            ########### dp
            # 计算群体0和群体1预测为正类的概率差值
            eval_dp_list = eval_dp_list / eval_data_count.sum(dim=1, keepdim=True)
            dp_positive = abs(eval_dp_list[0, 1] - eval_dp_list[1, 1])
            ########### deod
            eval_eod_list = eval_eod_list / (j+1) # 除以总batch数
            eval_max_eod = torch.max(eval_eod_list, dim=0)[0] - torch.min(eval_eod_list, dim=0)[0]
            eval_avg_eod = torch.mean(eval_max_eod).item() # 取tpr和fpr差距的均值
            eval_max_eod = torch.max(eval_max_eod).item()
            ########### deop
            eval_eopp_list = eval_eopp_list / eval_data_count
            eval_max_eopp = torch.max(eval_eopp_list, dim=0)[0] - torch.min(eval_eopp_list, dim=0)[0]
            eval_avg_eopp = torch.mean(eval_max_eopp).item()
            eval_max_eopp = torch.max(eval_max_eopp).item()

        model.train()
        eval_loss, eval_acc, eval_max_eopp, eval_avg_eopp, dp_positive, eval_avg_eod = self.scale_and_return(eval_loss, eval_acc, eval_max_eopp, eval_avg_eopp, dp_positive, eval_avg_eod)
        return eval_loss, eval_acc, eval_max_eopp, eval_avg_eopp, dp_positive, eval_avg_eod

    def save_model(self, save_dir, log_name="", model=None):
        model_to_save = self.model if model is None else model
        model_savepath = os.path.join(save_dir, log_name + '.pt')
        torch.save(model_to_save.state_dict(), model_savepath)

        print('Model saved to %s' % model_savepath)

    def compute_confusion_matix(self, dataset='test', num_classes=2,
                                dataloader=None, log_dir="", log_name=""):
        from scipy.io import savemat
        from collections import defaultdict
        self.model.eval()
        confu_mat = defaultdict(lambda: np.zeros((num_classes, num_classes)))
        print('# of {} data : {}'.format(dataset, len(dataloader.dataset)))

        predict_mat = {}
        output_set = torch.tensor([])
        group_set = torch.tensor([], dtype=torch.long)
        target_set = torch.tensor([], dtype=torch.long)
        intermediate_feature_set = torch.tensor([])

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # Get the inputs
                inputs, _, groups, targets, _ = data
                labels = targets
                groups = groups.long()

                if self.cuda:
                    inputs = inputs.cuda(self.device)
                    labels = labels.cuda(self.device)

                # forward

                outputs = self.model(inputs)
                if self.get_inter:
                    intermediate_feature = self.model.forward(inputs, get_inter=True)[-2]

                group_set = torch.cat((group_set, groups))
                target_set = torch.cat((target_set, targets))
                output_set = torch.cat((output_set, outputs.cpu()))
                if self.get_inter:
                    intermediate_feature_set = torch.cat((intermediate_feature_set, intermediate_feature.cpu()))

                pred = torch.argmax(outputs, 1)
                group_element = list(torch.unique(groups).numpy())
                for i in group_element:
                    mask = groups == i
                    if len(labels[mask]) != 0:
                        confu_mat[str(i)] += confusion_matrix(
                            labels[mask].cpu().numpy(), pred[mask].cpu().numpy(),
                            labels=[i for i in range(num_classes)])

        predict_mat['group_set'] = group_set.numpy()
        predict_mat['target_set'] = target_set.numpy()
        predict_mat['output_set'] = output_set.numpy()
        if self.get_inter:
            predict_mat['intermediate_feature_set'] = intermediate_feature_set.numpy()

        savepath = os.path.join(log_dir, log_name + '_{}_confu'.format(dataset))
        print('savepath', savepath)
        savemat(savepath, confu_mat, appendmat=True)

        savepath_pred = os.path.join(log_dir, log_name + '_{}_pred'.format(dataset))
        savemat(savepath_pred, predict_mat, appendmat=True)

        print('Computed confusion matrix for {} dataset successfully!'.format(dataset))
        return confu_mat
