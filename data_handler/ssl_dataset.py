"""
Original code:
    https://github.com/naver-ai/cgl_fairness
"""
from util_files.save_dis import *
from copy import deepcopy
import random
import numpy as np
import torch
from data_handler import GenericDataset
import os
import pickle
from util_files.ood_metrics import *
# 获取当前工作目录的绝对路径
current_working_directory = os.getcwd()
project_path = current_working_directory
import datetime
from utils import check_log_dir, make_log_name, set_seed
# 获取当前日期
current_date = datetime.date.today()
from util_files.maha_embedding_dis import *

# 计算权重的函数(ChatGPT想的)
def calculate_weights_gpt(dis_dict, alpha=1e-5):
    epsilon = 1e-5
    # 取所有dis值
    dis_values = np.array(list(dis_dict.values()))
    # 计算偏移量
    dis_min = np.min(dis_values)
    offset = np.abs(dis_min) + epsilon
    # 应用偏移量
    dis_offset = dis_values + offset
    # 归一化处理
    dis_normalized = dis_offset / np.max(dis_offset)
    # 计算权重
    weights = 1 / (dis_normalized + alpha)
    # 创建一个新的字典，其中包含样本ID和对应的权重
    weights_dict = {key: float(weight) for key, weight in zip(dis_dict.keys(), weights)}
    return weights_dict

# 计算权重的函数(w = exp(-γ*dis))
def calculate_weights_exp(dis_dict, gamma=0.5):
    weights_dict = {sample_id: np.exp(-gamma * dis) for sample_id, dis in dis_dict.items()}
    return weights_dict

# 计算权重的函数(w = (𝑑_𝑚𝑎𝑥−𝑑_𝑖 )^γ)
def calculate_weights_poly(dis_dict, gamma=0.5):
    # 取得所有距离值中的最大值
    d_max = max(dis_dict.values())
    # 根据公式计算每个样本的权重，并创建一个新的字典
    weights_dict = {sample_id: (d_max - dis)**gamma for sample_id, dis in dis_dict.items()}
    return weights_dict

class SSLDataset(GenericDataset):
    def __init__(self, sv_ratio=1.0, version='', ups_iter=0, **kwargs):
        super(SSLDataset, self).__init__(**kwargs)
        self.sv_ratio = sv_ratio
        self.version = version
        self.add_attr = None
        self.ups_iter = ups_iter

    def ssl_processing(self, features, num_data, idxs_per_group, idxs_dict=None, num_classes = 2, num_groups = 2):
        if self.sv_ratio >= 1:
            raise ValueError
        if self.split == 'test' and 'group' not in self.version:
            return features, num_data, idxs_per_group
        print('preprocessing for ssl...')
        num_groups, num_classes = num_data.shape
        folder_name = 'annotated_idxs'+f'_seed{self.args.seed}'
        # 说明：这个seed代表的是第几次run 实际上seed都是python内置随机种子生成器的结果
        idx_filename = '{}'.format(self.sv_ratio)
        if self.name == 'celeba':
            if self.target_attr != 'Attractive':
                idx_filename += f'_{self.target_attr}'
            if self.add_attr is not None:
                idx_filename += f'_{self.add_attr}'
        idx_filename += '.pkl'
        filepath = os.path.join(self.root, folder_name, idx_filename)
        if idxs_dict is None:
            if not os.path.isfile(filepath):
                idxs_dict = self.pick_idxs(num_data, idxs_per_group, filepath)
            else:
                with open(filepath, 'rb') as f:
                    idxs_dict = pickle.load(f)

        if self.version == 'groupclf_val':
            new_idxs = []
            val_idxs = []
            for g in range(num_groups):
                for l in range(num_classes):
                    if self.split == 'train':
                        train_num = int(len(idxs_dict['annotated'][(g, l)]) * 0.8)
                        idxs = idxs_dict['annotated'][(g, l)][:train_num]
                        val_idxs.extend(idxs_dict['annotated'][(g, l)][train_num:])
                    elif self.split == 'test':
                        idxs = idxs_dict['non-annotated'][(g, l)]
                    new_idxs.extend(idxs)
            new_idxs.sort()
            new_features = [features[idx] for idx in new_idxs]
            features = new_features
            self.val_idxs = val_idxs

        elif self.version == 'weight':
            folder = 'group_clf'
            model = 'resnet18' if self.name in ['utkface', 'celeba', 'utkface_fairface'] else 'mlp'
            epochs = '70' if self.name in ['utkface', 'celeba', 'utkface_fairface'] else '50'
            bs = '128' if self.name != 'adult' else '1024'
            # bs = self.args.img_size
            filename_pre = f'{model}_seed{self.args.seed}_epochs{epochs}_bs{bs}_lr0.001'
            filename_post = f'_sv{self.sv_ratio}_groupclf_val.pt'
            if self.name == 'celeba':
                if self.target_attr != 'Attractive':
                    filename_pre += f'_{self.target_attr}'
                if self.add_attr is not None:
                    filename_pre += f'_{self.add_attr}'
            filename = filename_pre + filename_post

            if self.name == 'utkface_fairface':
                path = os.path.join('./data/utkface_fairface', folder, filename)
            else:
                path = os.path.join(self.root, folder, filename)
            preds = torch.load(path)['pred']
            probs = torch.load(path)['probs']
            logits = torch.load(path)['logits']
            groups = torch.load(path)['group']
            thres = torch.load(path)['opt_thres']
            val_idxs = torch.load(path)['val_idxs']
            acc_list = torch.load(path)['acc_list']
            print('thres : ', thres)
            
            idx_pool = list(range(num_groups))
            total_per_group = np.zeros((num_classes, num_groups))
            for l in range(num_classes):
                for g in range(num_groups):
                    total_per_group[l, g] = len(idxs_dict['annotated'][(g, l)])
            total_per_group = total_per_group.astype(int)
            idx_anno = []
            idx_non = []
            for l in range(num_classes):
                for g in range(num_groups):
                    idx_anno.extend(idxs_dict['annotated'][(g, l)])
                    idx_non.extend(idxs_dict['non-annotated'][(g, l)])
            idx_gd_per_group = np.zeros((num_groups, num_classes), dtype=int)
            idx_hit = []
            idx_miss = []
            # 对于anno（含标签）数据，dis设置为最小值
            dis_dict_msp = {element: 0 for element in idx_anno}
            # 对于anno（含标签）数据，dis设置为最小值（注意：energy最小值要比non里面的最小值还小）
            dis_dict_energy = {} #后面要更新
            for g in range(num_groups):
                for l in range(num_classes):
                    idx_gd_per_group[g,l] = len(idxs_dict['annotated'][(g, l)])
                    for idx in idxs_dict['non-annotated'][(g, l)]:
                        prob_max = probs[idx].max()
                        ############# 使用pseudo方式判断是否打对 #############
                        features[idx][0] = preds[idx].item()
                        flag = (preds[idx] == groups[idx])
                        if flag:
                            idx_hit.append(idx)
                        else:
                            idx_miss.append(idx)
                        # MSP距离
                        msp = round(1-prob_max.item(),2)
                        dis_dict_msp[idx] = msp
            g0_acc = 0
            cnt1 = 0
            g1_acc = 0
            cnt2 = 0
            for l in range(num_classes):
                cnt1 += len(idxs_dict['non-annotated'][(0, l)])
                g0_acc += acc_list[idxs_dict['non-annotated'][(0, l)]].sum().item()
                cnt2 += len(idxs_dict['non-annotated'][(1, l)])
                g1_acc += acc_list[idxs_dict['non-annotated'][(1, l)]].sum().item()
            g0_acc = g0_acc / cnt1
            g1_acc = g1_acc / cnt2
            print("pseudo预测两个group的正确率:g0和g1")
            print(round(g0_acc,2), round(g1_acc,2))
            print("hit数量：")
            print(len(idx_hit))
            print("miss数量：")
            print(len(idx_miss))

            Y_logits = torch.tensor(np.array([logits[idx] for idx in idx_non]))
            G_Y = np.array([groups[idx] for idx in idx_non])
            G_pred = np.array([preds[idx] for idx in idx_non])
            G_prob = np.array([probs[idx] for idx in idx_non])

            dis_save_dir = os.path.join(f'{project_path}/distances', str(self.name), str(self.sv_ratio), str(self.args.seed))
            check_log_dir(dis_save_dir)
            weights_save_dir = os.path.join(f'{project_path}/weights', str(self.name), str(self.sv_ratio), str(self.args.seed))
            check_log_dir(weights_save_dir)
            dis_file_name = f'{dis_save_dir}/{self.args.dis_metric}_dis_json.json'
            # 定义要保存的列表变量
            # 检查数据是否已经存在
            distances = {}
            result = ''
            if not os.path.exists(dis_file_name):
                if 'energy' in self.args.dis_metric:
                    distances = energy_distance_with_indices(Y_logits, idx_non, G_pred, G_prob, G_Y)
                elif 'msp' in self.args.dis_metric:
                    distances = {key:dis_dict_msp[key] for key in idx_non}
                save_data(dis_file_name, distances)
                result = f"{self.args.dis_metric}_dis数据已保存到文件。"
            else:
                result = f"文件中已存在{self.args.dis_metric}_dis数据，无需重新计算，已取出该数据。"
                json_data = load_data(dis_file_name)
                # 使用字典推导式将所有的字符串键转换为整数键
                distances = {int(k): v for k, v in json_data.items()}
            # 初始化两个空列表
            dis_hit = []
            dis_miss = []
            # 根据idx_hit和idx_miss中的ID分配值到maha_hit和maha_miss
            for id in idx_hit:
                if id in distances:
                    dis_hit.append(distances[id])
            for id in idx_miss:
                if id in distances:
                    dis_miss.append(distances[id])
            dis_path = f'{dis_save_dir}/hit_miss_{self.args.dis_metric}.txt'
            if not os.path.exists(dis_path):
                with open(dis_path, 'w') as file:
                    for item in dis_hit:
                        file.write('{:.2f}  '.format(item))
                    file.write('\n')
                    for item in dis_miss:
                        file.write('{:.2f}  '.format(item))
                print(result)
                print("len(distances): ", str(len(distances)))
                print("dis与hit\miss相关内容已经存入文件！")

            ####################### 更新dis_dict_energy ######################
            if 'energy' in self.args.dis_metric:
                min_distance = min(distances.values())
                dis_dict_energy = {element: min_distance-1 for element in idx_anno}
                dis_dict_energy.update(distances)
                
            ####################### 权重设计反比于距离 #####################
            weights_path = f'{weights_save_dir}/weights_{self.args.dis_metric}_{self.args.w_method}_{self.args.gamma}.json'
            result = ''
            if not os.path.exists(weights_path):
                weights_dict = dict()
                if 'exp' in self.args.w_method:
                    if 'energy' in self.args.dis_metric:
                        weights_dict = calculate_weights_exp(dis_dict_energy, gamma=self.args.gamma)
                    elif 'msp' in self.args.dis_metric:
                        weights_dict = calculate_weights_exp(dis_dict_msp, gamma=self.args.gamma)
                    else:
                        raise ValueError
                elif 'poly' in self.args.w_method:
                    if 'energy' in self.args.dis_metric:
                        weights_dict = calculate_weights_poly(dis_dict_energy, gamma=self.args.gamma)
                    elif 'msp' in self.args.dis_metric:
                        weights_dict = calculate_weights_poly(dis_dict_msp, gamma=self.args.gamma)
                    else:
                        raise ValueError
                else:
                    raise ValueError
                weights_array_sorted = sorted(weights_dict.items(), key=lambda item: item[1])
                num_intervals = 5
                intervals = np.array_split(weights_array_sorted, num_intervals)
                new_weight_dict = {}
                # 将每个区间weight平均值该区间内所有样本的weight（不然fairhsic的bin方法weight_set搜索空间太大了）
                for interval in intervals:
                    mean_value = round(np.mean([float(item[1]) for item in interval]), 2)
                    for item in interval:
                        new_weight_dict[int(item[0])] = mean_value
                save_data(weights_path, new_weight_dict)
                result = f"{self.args.dis_metric}权重数据已保存到文件。"
            else:
                result = f"文件中已存在{self.args.dis_metric}权重数据。"
            print(result)
            
        elif self.version == 'cgl':
            folder = 'group_clf'
            model = 'resnet18' if self.name in ['utkface', 'celeba', 'utkface_fairface'] else 'mlp'
            epochs = '70' if self.name in ['utkface', 'celeba', 'utkface_fairface'] else '50'
            bs = '128' if self.name != 'adult' else '1024'
            filename_pre = f'{model}_seed{self.args.seed}_epochs{epochs}_bs{bs}_lr0.001'
            filename_post = f'_sv{self.sv_ratio}_groupclf_val.pt'
            if self.name == 'celeba':
                if self.target_attr != 'Attractive':
                    filename_pre += f'_{self.target_attr}'
                if self.add_attr is not None:
                    filename_pre += f'_{self.add_attr}'
            filename = filename_pre + filename_post

            if self.name == 'utkface_fairface':
                path = os.path.join('./data/utkface_fairface', folder, filename)
            else:
                path = os.path.join(self.root, folder, filename)

            preds = torch.load(path)['pred']
            probs = torch.load(path)['probs']
            thres = torch.load(path)['opt_thres']
            print('thres : ', thres)
            idx_pool = list(range(num_groups))

            total_per_group = np.zeros((num_classes, num_groups))
            for l in range(num_classes):
                for g in range(num_groups):
                    total_per_group[l, g] = len(idxs_dict['annotated'][(g, l)])
            total_per_group = total_per_group.astype(int)
            for g in range(num_groups):
                for l in range(num_classes):
                    for idx in idxs_dict['non-annotated'][(g, l)]:
                        if probs[idx].max() >= thres:
                            features[idx][0] = preds[idx].item()
                        else:
                            features[idx][0] = random.choices(idx_pool, k=1, weights=list(total_per_group[l]))[0]
        
        elif self.version == 'pseudo':
            folder = 'group_clf'
            model = 'resnet18' if self.name in ['utkface', 'celeba', 'utkface_fairface'] else 'mlp'
            epochs = '70' if self.name in ['utkface', 'celeba', 'utkface_fairface'] else '50'
            bs = '128' if self.name != 'adult' else '1024'
            filename_pre = f'{model}_seed{self.args.seed}_epochs{epochs}_bs{bs}_lr0.001'
            filename_post = f'_sv{self.sv_ratio}_groupclf_val.pt'
            if self.name == 'celeba':
                if self.target_attr != 'Attractive':
                    filename_pre += f'_{self.target_attr}'
                if self.add_attr is not None:
                    filename_pre += f'_{self.add_attr}'
            filename = filename_pre + filename_post

            if self.name == 'utkface_fairface':
                path = os.path.join('./data/utkface_fairface', folder, filename)
            else:
                path = os.path.join(self.root, folder, filename)

            preds = torch.load(path)['pred']
            probs = torch.load(path)['probs']
            thres = torch.load(path)['opt_thres']
            print('thres : ', thres)
            idx_pool = list(range(num_groups))

            total_per_group = np.zeros((num_classes, num_groups))
            for l in range(num_classes):
                for g in range(num_groups):
                    total_per_group[l, g] = len(idxs_dict['annotated'][(g, l)])
            total_per_group = total_per_group.astype(int)
            for g in range(num_groups):
                for l in range(num_classes):
                    for idx in idxs_dict['non-annotated'][(g, l)]:
                        features[idx][0] = preds[idx].item()
        elif self.version == 'remove':
            folder = 'group_clf'
            model = 'resnet18' if self.name in ['utkface', 'celeba', 'utkface_fairface'] else 'mlp'
            epochs = '70' if self.name in ['utkface', 'celeba', 'utkface_fairface'] else '50'
            bs = '128' if self.name != 'adult' else '1024'
            filename_pre = f'{model}_seed{self.args.seed}_epochs{epochs}_bs{bs}_lr0.001'
            filename_post = f'_sv{self.sv_ratio}_groupclf_val.pt'
            if self.name == 'celeba':
                if self.target_attr != 'Attractive':
                    filename_pre += f'_{self.target_attr}'
                if self.add_attr is not None:
                    filename_pre += f'_{self.add_attr}'
            filename = filename_pre + filename_post

            if self.name == 'utkface_fairface':
                path = os.path.join('./data/utkface_fairface', folder, filename)
            else:
                path = os.path.join(self.root, folder, filename)

            preds = torch.load(path)['pred']
            probs = torch.load(path)['probs']
            groups = torch.load(path)['group']
            thres = torch.load(path)['opt_thres']
            print('thres : ', thres)
            idx_pool = list(range(num_groups))

            total_per_group = np.zeros((num_classes, num_groups))
            for l in range(num_classes):
                for g in range(num_groups):
                    total_per_group[l, g] = len(idxs_dict['annotated'][(g, l)])
            total_per_group = total_per_group.astype(int)
            idx_miss = []
            for g in range(num_groups):
                for l in range(num_classes):
                    for idx in idxs_dict['non-annotated'][(g, l)]:
                        features[idx][0] = preds[idx].item()
                        flag = (preds[idx] == groups[idx])
                        if not flag:
                            idx_miss.append(idx)
            self.idx_miss = idx_miss
            # features = np.delete(features, idx_miss, axis=0) # 去掉敏感属性打错的
            # print('count the number of data newly!')
            num_data, idxs_per_group = self._data_count(features, num_groups, num_classes)
            return features, num_data, idxs_per_group
        
        elif self.version == 'dl':
            idx_non = []
            for l in range(num_classes):
                for g in range(num_groups):
                    idx_non.extend(idxs_dict['non-annotated'][(g, l)])
            print("len(features) before:")
            print(len(features))
            features = np.delete(features, idx_non, axis=0) # 仅使用DL进行训练
            print("len(features) after:")
            print(len(features))
            print('count the number of data newly!')
            num_data, idxs_per_group = self._data_count(features, num_groups, num_classes)
            return features, num_data, idxs_per_group
        
        elif self.version == 'fairif_pseudo':
            # fair_if要用
            dl_idxs = []
            du_idxs = []
            for g in range(num_groups):
                for l in range(num_classes):
                    if self.split == 'train':
                        idxs1 = idxs_dict['annotated'][(g, l)]
                        dl_idxs.extend(idxs1)
                        idxs2 = idxs_dict['non-annotated'][(g, l)]
                        du_idxs.extend(idxs2)
            self.dl_idxs = dl_idxs
            self.du_idxs = du_idxs
            print("ssl:dl_idx的数量：{}".format(len(dl_idxs)))
            print("ssl:du_idx的数量：{}".format(len(du_idxs)))
            # print(dl_idxs)
        else:
            raise ValueError
        
        print("======================OK!权重计算和存储结束！========================")
        print('count the number of data newly!')
        num_data, idxs_per_group = self._data_count(features, num_groups, num_classes)
        return features, num_data, idxs_per_group

    def pick_idxs(self, num_data, idxs_per_group, filepath):
        print('<pick idxs : {}>'.format(filepath))
        if not os.path.isdir(os.path.join(self.root, 'annotated_idxs'+f'_seed{self.args.seed}')):
            os.mkdir(os.path.join(self.root, 'annotated_idxs'+f'_seed{self.args.seed}'))
        num_groups, num_classes = num_data.shape
        idxs_dict = {}
        idxs_dict['annotated'] = {}
        idxs_dict['non-annotated'] = {}
        for g in range(num_groups):
            for l in range(num_classes):
                num_nonannotated = int(num_data[g, l] * (1-self.sv_ratio))
                print(g, l, num_nonannotated)
                # random.seed(self.args.seed) # cyl
                idxs_nonannotated = random.sample(idxs_per_group[(g, l)], num_nonannotated)
                idxs_annotated = [idx for idx in idxs_per_group[(g, l)] if idx not in idxs_nonannotated]
                idxs_dict['non-annotated'][(g, l)] = idxs_nonannotated
                idxs_dict['annotated'][(g, l)] = idxs_annotated
        with open(filepath, 'wb') as f:
            pickle.dump(idxs_dict, f, pickle.HIGHEST_PROTOCOL)
        return idxs_dict
    
    def pick_idxs_ood(self, num_data, idxs_per_group, filepath):
        print('<pick idxs : {}>'.format(filepath))
        if not os.path.isdir(os.path.join(self.root, 'annotated_idxs_ood'+f'_seed{self.args.seed}')):
            os.mkdir(os.path.join(self.root, 'annotated_idxs_ood'+f'_seed{self.args.seed}'))
        num_groups, num_classes = num_data.shape
        idxs_dict = {}
        idxs_dict['annotated'] = {}
        idxs_dict['non-annotated'] = {}
        for g in range(num_groups): # 只取0这一个group作为annotated
            for l in range(num_classes):
                num_annotated = int(num_data[g, l] * (self.sv_ratio)*2)
                if g == 1:
                    num_annotated = 0
                print(g, l, num_annotated)
                random.seed(self.args.seed) # cyl
                idxs_annotated = []
                idxs_annotated = random.sample(idxs_per_group[(g, l)], num_annotated)
                idxs_nonannotated = [idx for idx in idxs_per_group[(g, l)] if idx not in idxs_annotated]
                idxs_dict['non-annotated'][(g, l)] = idxs_nonannotated
                idxs_dict['annotated'][(g, l)] = idxs_annotated
        with open(filepath, 'wb') as f:
            pickle.dump(idxs_dict, f, pickle.HIGHEST_PROTOCOL)
        return idxs_dict
