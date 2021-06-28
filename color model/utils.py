
import torch
import scipy.io as scio
import numpy as np


def generate_masks(mask_path):
    mask = scio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask = np.transpose(mask, [2, 0, 1])
    mask_s = np.sum(mask, axis=0)
    index = np.where(mask_s == 0)
    mask_s[index] = 1
    mask_s = mask_s.astype(np.uint8)
    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask = mask.cuda()
    mask_s = torch.from_numpy(mask_s)
    mask_s = mask_s.float()
    mask_s = mask_s.cuda()
    return mask, mask_s


def split_masks(mask, scale, args):
    mask_list = list()
    for i in range(scale):
        for j in range(scale):
            if len(mask.shape) == 3:
                mask_list.append(mask[:, j * args.size[0] // scale:(j + 1) * args.size[0] // scale,
                                 i * args.size[1] // scale:(i + 1) * args.size[1] // scale])

            elif len(mask.shape) == 2:
                mask_list.append(mask[j * args.size[0] // scale:(j + 1) * args.size[0] // scale,
                                 i * args.size[1] // scale:(i + 1) * args.size[1] // scale])
            elif len(mask.shape) == 4:
                mask_list.append(mask[:, :, j * args.size[0] // scale:(j + 1) * args.size[0] // scale,
                                 i * args.size[1] // scale:(i + 1) * args.size[1] // scale])

    return mask_list


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename
