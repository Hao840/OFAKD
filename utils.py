import time
from datetime import datetime

import numpy as np
from timm.data import ImageDataset
from torchvision.datasets import CIFAR100


class ImageNetInstanceSample(ImageDataset):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, name, class_map, load_bytes, is_sample=False, k=4096, **kwargs):
        super().__init__(root, parser=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 1000
            num_samples = len(self.parser)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.parser[i]
                label[i] = target

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = super().__getitem__(index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class CIFAR100InstanceSample(CIFAR100, ImageNetInstanceSample):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, train, is_sample=False, k=4096, **kwargs):
        CIFAR100.__init__(self, root, train, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 100
            num_samples = len(self.data)

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[self.targets[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        img, target = CIFAR100.__getitem__(self, index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class TimePredictor:
    def __init__(self, steps, most_recent=30, drop_first=True):
        self.init_time = time.time()
        self.steps = steps
        self.most_recent = most_recent
        self.drop_first = drop_first  # drop iter 0

        self.time_list = []
        self.temp_time = self.init_time

    def update(self):
        time_interval = time.time() - self.temp_time
        self.time_list.append(time_interval)

        if self.drop_first and len(self.time_list) > 1:
            self.time_list = self.time_list[1:]
            self.drop_first = False

        self.time_list = self.time_list[-self.most_recent:]
        self.temp_time = time.time()

    def get_pred_text(self):
        single_step_time = np.mean(self.time_list)
        end_timestamp = self.init_time + single_step_time * self.steps
        return datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')
