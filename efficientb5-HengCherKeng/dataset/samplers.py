import torch
import numpy


class FiveBalanceClassSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        label = self.dataset.dataframe["Label"].values
        label = label.reshape(-1, 4)
        label = numpy.hstack([label.sum(1, keepdims=True) == 0, label]).T
        self.neg_index = numpy.where(label[0])[0]
        self.pos1_index = numpy.where(label[1])[0]
        self.pos2_index = numpy.where(label[2])[0]
        self.pos3_index = numpy.where(label[3])[0]
        self.pos4_index = numpy.where(label[4])[0]
        # 5x
        self.num_image = len(self.dataset.dataframe) // 4
        self.length = self.num_image * 5

    def __iter__(self):
        neg = numpy.random.choice(self.neg_index, self.num_image, replace=True)
        pos1 = numpy.random.choice(self.pos1_index, self.num_image, replace=True)
        pos2 = numpy.random.choice(self.pos2_index, self.num_image, replace=True)
        pos3 = numpy.random.choice(self.pos3_index, self.num_image, replace=True)
        pos4 = numpy.random.choice(self.pos4_index, self.num_image, replace=True)
        l = numpy.stack([neg, pos1, pos2, pos3, pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length


class FixedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = index
        self.length = len(index)

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return self.length


class FixedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, length=-1):
        self.dataset = dataset
        if length < 0:
            length = len(dataset)
        self.length = length

    def __iter__(self):
        dataset_length = len(self.dataset)
        index = numpy.random.choice(dataset_length, self.length, replace=True)
        return iter(index)

    def __len__(self):
        return self.length
