from __future__ import division, print_function
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

# credits to https://stackoverflow.com/questions/11955000/how-to-preserve-matlab-struct-when-accessing-in-python


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def normalization_v1(input):
    """ Python implementation of NormalizeFea.m """
    # be aware that the input data has to be N * d, rows represent number of samples
    # nSmp = input.shape[0]
    feaNorm = np.maximum(1e-14, pow(input, 2).sum(1))
    feaNorm = np.diag(pow(feaNorm, -0.5))
    return np.matmul(feaNorm, input)


def normalization_v2(input, unitNorm=True):
    """ mean 0 std 1, with unit norm """
    # be aware that the input data has to be N * d, rows represent number of samples
    sampleMean = np.mean(input, axis=1).reshape(
        input.shape[0], 1)  # reshape the vector to a N * 1 matrix
    sampleStd = np.std(input, axis=1).reshape(input.shape[0], 1)

    # take advantage of the broadcasting operation in Numpy
    input = (input - sampleMean) / sampleStd
    sampleNorm = np.linalg.norm(input, axis=1).reshape(input.shape[0], 1)

    # transform to unit norm
    if unitNorm:
        input = input / sampleNorm

    return input


def load_cancer_data(fileName, path=None):
    print(path)
    filePath = path + '/' + fileName
    print(filePath)
    # get file name
    filePrefix = fileName.split('.')[0]
    originData = _check_keys(sio.loadmat(
        filePath, struct_as_record=False, squeeze_me=True))
    # return dict([(filePrefix, data)]) # or simply dict.fromkeys(key1, data)
    # print('debugging')

    data = []
    view_names = ['exp', 'mirna', 'methy']
    for vname in view_names:
        curData = np.array(list(originData[vname].values()))
        # normalize the data
        # we require the output to be a list of tensors
        data.append(torch.from_numpy(normalization_v2(curData)).float())
    # data = torch.stack(curData, dim=2) # different dimensions of each view, cannot be stacked into a tensor

    labels = list(originData[view_names[0]].keys())

    print(f'Number of views:{len(data)}\nNumber of samples:{len(labels)}')

    return data, labels


class CancerDataset(Dataset):
    def __init__(self, fileName, path):

        self.data, self.labels = load_cancer_data(fileName, path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # for temp in self.data:
        #     curData.append(temp[idx])
        #
        # 将idx编号的数据提取出来(注意是对应的每个视图)
        curData = [temp[idx] for temp in self.data]
        # 输出视图的对应的数据,对应的标签,标签所对应的数字
        return curData, self.labels[idx], torch.from_numpy(np.array(idx))


# The following code is for debugging purpose only
data = load_cancer_data('aml.mat', '/Users/sky/Desktop/CC-Mine/TCGA/data')
print('success')
dataset = CancerDataset('aml.mat', '/Users/sky/Desktop/CC-Mine/TCGA/data')
train_loader = DataLoader(dataset, batch_size=128, shuffle=False)
print('--------------------------------------------')
for batch_idx, (x, _, idx) in enumerate(train_loader):
    if batch_idx == 0:
        print(x)
        print('--------------------------------------------')
        print(x[0][0])
        print(x[0][1])
        print(x[0][2])

    print(f'Current batch : {batch_idx}')
