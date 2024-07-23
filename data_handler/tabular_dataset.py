"""
Original code:
    https://github.com/naver-ai/cgl_fairness
"""
from collections import defaultdict
import numpy as np
import random
from data_handler import SSLDataset

class TabularDataset(SSLDataset):
    """Adult dataset."""
    # 1 idx -> sensi
    # 2 idx -> label
    # 3 idx -> filename or feature (image / tabular)
    def __init__(self, dataset, sen_attr_idx, **kwargs):
        super(TabularDataset, self).__init__(**kwargs)
        self.sen_attr_idx = sen_attr_idx
        dataset_train, dataset_test = dataset.split([0.8], shuffle=False, seed=0)
        # dataset_train, dataset_test = dataset.split([0.8], shuffle=True, seed=5)
        # features, labels = self._balance_test_set(dataset)
        self.dataset = dataset_train if (self.split == 'train') \
        or ('group' in self.version) else dataset_test

        features = np.delete(self.dataset.features, self.sen_attr_idx, axis=1)
        mean, std = self._get_mean_n_std(dataset_train.features)
        features = (features - mean) / std

        self.groups = np.expand_dims(self.dataset.features[:, self.sen_attr_idx], axis=1)
        self.labels = np.squeeze(self.dataset.labels)

        # self.features = self.dataset.features
        self.features = np.concatenate((self.groups, self.dataset.labels, features), axis=1)

        # For prepare mean and std from the train dataset
        self.num_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)
        '''
        # if (self.split == 'train') or ('group' in self.version):
        #     self.num_data, self.idxs_per_group = self.undersample_data_and_counts(self.num_data, self.idxs_per_group)
        #     print("下采样后：")
        #     for i in range(self.num_groups):
        #         print('# of %d group data : ' % i,self.num_data[i, :])
        # if semi-supervised learning,
        '''
        print("===============真的没有进入ssl呀！！！！！！！！！！！！！！")
        if self.sv_ratio < 1:
            print("============啊这 进入ssl了！============")
            # we want the different supervision according to the seed
            random.seed(self.seed)
            self.features, self.num_data, self.idxs_per_group = self.ssl_processing(self.features, self.num_data, self.idxs_per_group, None, self.num_classes, self.num_groups)
            if 'group' in self.version:
                a, b = self.num_groups, self.num_classes
                self.num_groups, self.num_classes = b, a

    def get_dim(self):
        return self.dataset.features.shape[-1]

    def __getitem__(self, idx):
        features = self.features[idx]
        group = features[0]
        label = features[1]
        feature = features[2:]
        if 'group' in self.version:
            return np.float32(feature), 0, label, np.int64(group), (idx, 0)
        else:
            return np.float32(feature), 0, group, np.int64(label), (idx, 0)

    def _get_mean_n_std(self, train_features):
        features = np.delete(train_features, self.sen_attr_idx, axis=1)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] += 1e-7
        return mean, std

    def undersample_data_and_counts(data_count, idxs_per_group):
        """
        根据每个组和类别的样本数量进行下采样，并返回下采样后的样本数。
        :param data_count: numpy数组，包含每个组和类别的样本数。
        :param idxs_per_group: 字典，每个组和类别的样本索引列表。
        :return: 下采样后的样本索引字典及每个组和类别下采样后的样本数。
        """
        # 确定每个组合中的最小样本数
        min_samples = data_count.min()
        
        # 创建一个新的下采样字典和样本数量数组
        undersampled_idxs = defaultdict(lambda: defaultdict(list))
        undersampled_counts = np.zeros_like(data_count)
        
        # 对每个组和类别进行下采样
        for group in idxs_per_group:
            for class_ in idxs_per_group[group]:
                # 随机选择min_samples数量的样本，无重复
                undersampled_idxs[group][class_] = np.random.choice(
                    idxs_per_group[group][class_], min_samples, replace=False).tolist()
                # 更新下采样后的样本数量
                undersampled_counts[group, class_] = min_samples
        
        return undersampled_counts, undersampled_idxs

    def undersample_data_and_counts(self, data_count, idxs_per_group):
        """
        根据每个组和类别的样本数量进行下采样，并返回下采样后的样本数和样本索引。
        
        :param data_count: numpy数组，包含每个组和类别的样本数。
        :param idxs_per_group: 字典，每个组和类别的样本索引列表。
        :return: 下采样后的样本索引字典及每个组和类别下采样后的样本数。
        """
        print(idxs_per_group)
        # 确定每个组合中的最小样本数
        min_samples = data_count.min()
        
        # 创建一个新的下采样字典和样本数量数组
        undersampled_idxs = defaultdict(lambda: defaultdict(list))
        undersampled_counts = np.zeros_like(data_count)
        
        # 对每个组和类别进行下采样
        for group in range(data_count.shape[0]):
            for class_ in range(data_count.shape[1]):
                # 检查当前组和类别的索引列表是否为空
                if len(idxs_per_group[(group,class_)]) >= min_samples:
                    # 随机选择min_samples数量的样本，无重复
                    undersampled_idxs[(group,class_)] = np.random.choice(
                        idxs_per_group[(group,class_)], min_samples, replace=False).tolist()
                    # 更新下采样后的样本数量
                    undersampled_counts[group, class_] = min_samples
                else:
                    # 如果索引列表为空，不进行抽样，记录为0
                    undersampled_idxs[(group,class_)] = []
                    undersampled_counts[group, class_] = 0
        
        return undersampled_counts, undersampled_idxs