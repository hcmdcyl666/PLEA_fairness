"""
Original code:
    https://github.com/naver-ai/cgl_fairness
"""
import argparse
import os
import sys
# 获取当前工作目录的绝对路径
current_working_directory = os.getcwd()
project_path = current_working_directory
import datetime

# 获取当前日期
current_date = datetime.date.today()

def get_args():
    parser = argparse.ArgumentParser(description='Fairness')
    parser.add_argument('--w-method', default='exp', type=str, help='method for w', choices=['exp', 'poly'])
    parser.add_argument('--gamma', default=0.5, type=float, help='parameter for w', choices=[0.5, 1.0, 2.0])
    # parser.add_argument('--dis-method', default='bin', type=str, help='dis method', choices=['bin', 'MC'])
    parser.add_argument('--dis-method', type=str, help='dis method')
    parser.add_argument('--dis-metric', default='msp', type=str, help='dis metric', choices=['energy', 'msp', 'prob'])
    parser.add_argument('--num_workers', default=1, type=int, help='num of workers')
    parser.add_argument('--t', default=1, type=int, help='temperature')
    parser.add_argument('--date1', default=f'{current_date}',
                        type=str, help='experiment date')
    parser.add_argument('--result-dir', default=f'{project_path}/results/',
                        help='directory to save results (default: project/results/)')
    parser.add_argument('--date', default='20200101', type=str, help='experiment date')
    parser.add_argument('--term', default=20, type=int, help='the period for recording train acc')
    parser.add_argument('--record', default=False, action='store_true', help='record using tensorboardX')
    # parser.add_argument('--result-dir', default='./results/',
    #                     help='directory to save results (default: ./results/)')
    parser.add_argument('--log-dir', default='./logs/',
                        help='directory to save logs (default: ./logs/)')
    parser.add_argument('--data-dir', default='./data/',
                        help='data directory (default: ./data/)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save trained models (default: ./trained_models/)')
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--evalset', default='test', choices=['all', 'train', 'test'])
    parser.add_argument('--get-inter', default=False, action='store_true',
                        help='get penultimate features for TSNE visualization')

    #### base configuration for learning ####
    parser.add_argument('--seed', default=1, type=int, help='seed run for randomness')
    # dataset
    parser.add_argument('--dataset', required=True, default='compas', choices=['adult', 'compas', 'german', 'utkface', 'celeba', 'utkface_fairface'])
    parser.add_argument('--batch-size', default=128, type=int, help='mini batch size')
    parser.add_argument('--img-size', default=224, type=int, help='img size for preprocessing')
    parser.add_argument('--num-workers', default=2, type=int, help='the number of thread used in dataloader')
    parser.add_argument('--labelwise', default=False, action='store_true', help='balanced sampling over groups')
    # only for celebA
    parser.add_argument('--target', default='Attractive', type=str, help='target attribute for celeba')
    parser.add_argument('--add-attr', default=None, help='additional group attribute for celeba')
    # model
    parser.add_argument('--model', default='mlp', required=True, choices=['mlp', 'resnet18','resnet18_dropout'])
    parser.add_argument('--modelpath', default=None)
    parser.add_argument('--group-modelpath', default=None)
    parser.add_argument('--bg-modelpath', default=None)
    parser.add_argument('--pretrained', default=False, action='store_true', help='load imagenet pretrained model')
    parser.add_argument('--device', default=0, type=int, help='cuda device number')
    parser.add_argument('--t-device', default=0, type=int, help='teacher cuda device number')
    # optimization
    parser.add_argument('--method', default='scratch', type=str, required=True,
                        choices=['scratch', 'reweighting','mfd','mfd_cgl', 'adv', 'fairif', 'fairhsic', 'fairhsic_bin', 'mfd_bin', 'fairhsic_cgl'])
    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                        choices=['AdamP', 'SGD', 'SGD_momentum_decay', 'Adam'],
                        help='(default=%(default)s)')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=0, type=float, help='weight decay')


    # for base fairness methods
    parser.add_argument('--lamb', default=1, type=float, help='fairness strength')
    # for MFD
    parser.add_argument('--sigma', default=1.0, type=float, help='sigma for rbf kernel')
    parser.add_argument('--kernel', default='rbf', type=str, choices=['rbf', 'poly'], help='kernel for mmd')
    parser.add_argument('--teacher-type', default=None, choices=['mlp', 'resnet18', 'resnet18_dropout'])
    parser.add_argument('--teacher-path', default=None, help='teacher model path')

    # For reweighting & adv
    parser.add_argument('--reweighting-target-criterion', default='eo', type=str, help='fairness criterion')
    parser.add_argument('--iteration', default=10, type=int, help='iteration for reweighting')
    parser.add_argument('--ups-iter', default=10, type=int, help='iteration for reweighting')
    parser.add_argument('--eta', default=0.001, type=float, help='adversary training learning rate or lr for reweighting')

    # For fair FG,
    parser.add_argument('--sv', default=1, type=float, help='the ratio of group annotation for a training set')
    parser.add_argument('--version', default='', type=str, help='version about how the unsupervised data is used')

    args = parser.parse_args()
    args.cuda=True
    if args.mode == 'train' and args.method == 'mfd':
        if args.teacher_type is None:
            raise Exception('A teacher model needs to be specified for distillation')
        elif args.teacher_path is None:
            raise Exception('A teacher model path is not specified.')

    return args
