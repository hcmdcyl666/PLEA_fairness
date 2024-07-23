"""
Original code:
    https://github.com/Trusted-AI/AIF360
"""
import os
import pandas as pd
from data_handler.AIF360.standard_dataset import StandardDataset

default_mappings = {
    # 'label_maps': [{1.0: 'Good Credit', 2.0: 'Bad Credit'}],
    'label_maps': [{1.0: 1, 0.0: 2}], # 在原始数据中，1表示good credit，2表示bad credit
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
                                 {1.0: 'Old', 0.0: 'Young'}],
}

class GermanDataset(StandardDataset):
    """German credit Dataset.

    See :file:`aif360/data/raw/german/README.md`.
    """

    def __init__(self, data, label_name='credit', favorable_classes=[1],
                 protected_attribute_names=['sex', 'age'],
                 privileged_classes=[['male'], lambda x: x > 25],
                 instance_weights_name=None,
                 categorical_features=['status', 'credit_history', 'purpose',
                     'savings', 'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone',
                     'foreign_worker'],
                 features_to_keep=[], features_to_drop=['personal_status'],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments.

        By default, this code converts the 'age' attribute to a binary value
        where privileged is `age > 25` and unprivileged is `age <= 25` as
        proposed by Kamiran and Calders [1]_.

        References:
            .. [1] F. Kamiran and T. Calders, "Classifying without
               discriminating," 2nd International Conference on Computer,
               Control and Communication, 2009.

        Examples:
            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:

            >>> label_map = {1.0: 'Good Credit', 0.0: 'Bad Credit'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> gd = GermanDataset(protected_attribute_names=['sex'],
            ... privileged_classes=[['male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})

            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """

        
        super(GermanDataset, self).__init__(df=data, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
