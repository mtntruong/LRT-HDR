import os
import cv2
import h5py
import copy
import OpenEXR
import Imath
import numpy as np
import torch
from torch.utils.data import Dataset


def read_EXR(filename):

    exr = OpenEXR.InputFile(filename)
    header = exr.header()

    dw = header['dataWindow']
    img_size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exr.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, img_size)
        channelData[c] = C

    exr.close()
    return channelData


def write_EXR(filename, hdr_data):
    height, width, depth = hdr_data.shape
    channel_names = ['R', 'G', 'B']
    header = OpenEXR.Header(width, height)
    header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    header['channels'] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for c in channel_names}
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels({c: hdr_data[:, :, i].astype(np.float32).tostring() for i, c in enumerate(channel_names)})
    out.close()


def luma(input):

    a = 17.554; b = 826.81; c = 0.10013; d = -884.17; e = 209.16; f = -731.28
    yl = 5.6046; yh = 10469

    output = copy.deepcopy(input)
    output[input < yl] = a * input[input < yl]
    output[(input >= yl) & (input < yh)] = b * np.power(input[(input >= yl) & (input < yh)], c) + d
    output[input >= yh] = e * np.log(input[input >= yh]) + f
    output = output / 4096.0

    return output


def inv_luma(input):

    a = 0.056968; b = 7.3014e-30; c = 9.9872; d = 884.17; e = 32.994; f = 0.0047811
    ll = 98.381; lh = 1204.7

    new_input = copy.deepcopy(input)
    new_input = new_input * 4096.0
    output = copy.deepcopy(new_input)
    output[new_input < ll] = a * new_input[new_input < ll]
    output[(new_input >= ll) & (new_input < lh)] = b * np.power((new_input[(new_input >= ll) & (new_input < lh)] + d), c)
    output[new_input >= lh] = e * np.exp(f * new_input[new_input >= lh])

    return output


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h

    return h


class ExpandNetLoss(torch.nn.Module):
    def __init__(self, loss_lambda=5):
        super(ExpandNetLoss, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l1_loss = torch.nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        cosine_term = (torch.tensor(1) - self.similarity(x, y)).mean()
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term


class DatasetDirectory(Dataset):
    def __init__(self, path):
        self.gt_path = os.path.join(path, 'GT')
        self.omega_path = os.path.join(path, 'OMEGA')
        self.seq_path = os.path.join(path, 'SEQ')
        self.mat_list = os.listdir(self.gt_path)

    def __getitem__(self, index):
        f = h5py.File(os.path.join(self.seq_path, self.mat_list[index]), 'r')
        reader = f.get('radiance_seq')
        data = torch.from_numpy(np.array(reader).astype('float32'))

        f = h5py.File(os.path.join(self.omega_path, self.mat_list[index]), 'r')
        reader = f.get('omega')
        omega = torch.from_numpy(np.array(reader).astype('float32'))

        f = h5py.File(os.path.join(self.gt_path, self.mat_list[index]), 'r')
        reader = f.get('radiance')
        target = torch.from_numpy(np.array(reader).astype('float32'))

        return data, omega, target

    def __len__(self):
        return len([1 for x in list(os.scandir(self.gt_path)) if x.is_file()])
