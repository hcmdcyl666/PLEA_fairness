"""
Original code:
    https://github.com/sangwon79/Fair-Feature-Distillation-for-Visual-Recognition
"""
from data_handler.dataset_factory import DatasetFactory
import torch
import torch.optim as optim
import numpy as np
import networks
import data_handler
from torch.utils.data import DataLoader
import trainer
from utils import check_log_dir, make_log_name, set_seed, get_samples_by_indices
# from adamp import AdamP
# from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networks
import torch.nn.functional as F
from arguments import get_args
import time
import os
args = get_args()

def get_weights(loader, cuda=True):
    num_groups = loader.dataset.num_groups
    data_counts = torch.zeros(num_groups)
    data_counts = data_counts.cuda() if cuda else data_counts
    for data in loader:
        inputs, _, groups, _, _ = data
        for g in range(num_groups):
            data_counts[g] += torch.sum((groups == g))

    weights = data_counts / data_counts.min()
    return weights, data_counts

def focal_loss(input_values, gamma=10):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.5):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

def main():
    torch.backends.cudnn.enabled = True

    seed = args.seed
    set_seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    log_name = make_log_name(args)
    dataset = args.dataset
    save_dir = os.path.join(args.save_dir, args.date, dataset, args.method)
    result_dir = os.path.join(args.result_dir, args.date, dataset, args.method)
    check_log_dir(save_dir)
    check_log_dir(result_dir)
    writer = None
    # if args.record:
    #     log_dir = os.path.join(args.log_dir, args.date, dataset, args.method)
    #     check_log_dir(log_dir)
    #     writer = SummaryWriter(log_dir + '/' + log_name)
    ########################## get dataloader ################################
    if args.dataset == 'adult':
        args.img_size = 97
    elif args.dataset == 'compas':
        args.img_size = 400
    else:
        args.img_size = 224
    print("args.sv :")
    print(str(args.sv))
    tmp = data_handler.DataloaderFactory.get_dataloader(args.dataset,
                                                        batch_size=args.batch_size, seed=args.seed,
                                                        num_workers=args.num_workers,
                                                        target_attr=args.target,
                                                        add_attr=args.add_attr,
                                                        labelwise=args.labelwise,
                                                        sv_ratio=args.sv,
                                                        version=args.version,
                                                        args=args
                                                        )
    num_classes, num_groups, train_loader, test_loader = tmp
    # print("数据处理完了 先结束！")
    # return
    ########################## get model ##################################
    if args.dataset == 'adult':
        args.img_size = 97
    elif args.dataset == 'compas':
        args.img_size = 400
    elif 'cifar' in args.dataset:
        args.img_size = 32
    # -2是因为group和class是feature之外的
    args.img_size = len(train_loader.dataset.features[0])-2
    print("args.img_size:")
    print(args.img_size)
    model = networks.ModelFactory.get_model(args.model, num_classes, args.img_size,
                                            pretrained=args.pretrained, num_groups=num_groups)

    model.cuda('cuda:{}'.format(args.device))

    if args.modelpath is not None:
        model.load_state_dict(torch.load(args.modelpath))

    teacher = None
    if (args.method == 'mfd' or args.teacher_path is not None) and args.mode != 'eval':
        teacher = networks.ModelFactory.get_model(args.teacher_type, train_loader.dataset.num_classes, args.img_size)
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=torch.device('cuda:{}'.format(args.t_device))))
        teacher.cuda('cuda:{}'.format(args.t_device))

    ########################## get trainer ##################################
    if 'Adam' in args.optimizer:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # elif 'AdamP' in args.optimizer:
    #     optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif 'SGD' in args.optimizer:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    trainer_ = trainer.TrainerFactory.get_trainer(args.method, model=model, args=args,
                                                  optimizer=optimizer, teacher=teacher)

    ####################### start training or evaluating ####################
    if args.mode == 'train':
        start_t = time.time()
        # dl_idxs = train_loader.dataset.dl_idxs
        # _, du_loader = get_samples_by_indices(train_loader, dl_idxs)
        trainer_.train(train_loader, test_loader, args.epochs, writer=writer)
        
        # fair_if
        # dl_idxs = train_loader.dataset.dl_idxs
        # du_idxs = train_loader.dataset.du_idxs
        # trainer_.train(train_loader, dl_idxs, du_idxs, test_loader, args.epochs, writer=writer) # fair_if需要val数据
        end_t = time.time()
        train_t = int((end_t - start_t)/60)  # to minutes
        print('Training Time : {} hours {} minutes'.format(int(train_t/60), (train_t % 60)))
        print("target classifier train完存入模型！")
        trainer_.save_model(save_dir, log_name)
    else:
        print('Evaluation ----------------')
        model_to_load = args.modelpath
        trainer_.model.load_state_dict(torch.load(model_to_load))
        print("target classifier eval完毕！")
        print('Trained model loaded successfully')

    print('Done!')


if __name__ == '__main__':
    main()
