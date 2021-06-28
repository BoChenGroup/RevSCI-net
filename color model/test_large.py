
from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from models import  re_3dcnn1
from utils import generate_masks, time2file_name, split_masks
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import argparse
import datetime
import os
import numpy as np
from torch.autograd import Variable

# from thop import profile

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

data_path = "./largescale_rgb"
test_path1 = "./test"

parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')
parser.add_argument('--last_train', default=20, type=int, help='pretrain model')
parser.add_argument('--model_save_filename', default='color_model', type=str,
                    help='pretrain model save folder name')
parser.add_argument('--max_iter', default=100, type=int, help='max epoch')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--B', default=24, type=int, help='compressive rate')
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--size', default=[1080, 1920], type=int, help='input image resolution')
parser.add_argument('--mode', default='noreverse', type=str, help='training mode: reverse or noreverse')
args = parser.parse_args()

mask, mask_s = generate_masks(data_path)


loss = nn.MSELoss()
loss.cuda()


def test(test_path, epoch, result_path, model, args):
    r = np.array([[1, 0], [0, 0]])
    g1 = np.array([[0, 1], [0, 0]])
    g2 = np.array([[0, 0], [1, 0]])
    b = np.array([[0, 0], [0, 1]])
    rgb2raw = np.zeros([3, args.size[0], args.size[1]])
    rgb2raw[0, :, :] = np.tile(r, (args.size[0] // 2, args.size[1] // 2))
    rgb2raw[1, :, :] = np.tile(g1, (args.size[0] // 2, args.size[1] // 2)) + np.tile(g2, (
        args.size[0] // 2, args.size[1] // 2))
    rgb2raw[2, :, :] = np.tile(b, (args.size[0] // 2, args.size[1] // 2))
    rgb2raw = torch.from_numpy(rgb2raw).cuda().float()

    test_list = os.listdir(test_path)
    psnr_cnn = torch.zeros(len(test_list))
    for i in range(len(test_list)):
        pic = scio.loadmat(test_path + '/' + test_list[i])

        if "orig" in pic:
            pic = pic['orig']
        elif "patch_save" in pic:
            pic = pic['patch_save']
        pic = pic / 255

        pic_gt = np.zeros([pic.shape[3] // args.B, args.B, 3, args.size[0], args.size[1]])
        for jj in range(pic.shape[3]):
            if jj % args.B == 0:
                meas_t = np.zeros([args.size[0], args.size[1]])
                n = 0
            pic_t = pic[:, :, :, jj]
            mask_t = mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_t = np.transpose(pic_t, [2, 0, 1])
            pic_gt[jj // args.B, n, :, :, :] = pic_t

            n += 1
            meas_t = meas_t + np.multiply(mask_t.numpy(), torch.sum(torch.from_numpy(pic_t).cuda().float() * rgb2raw,
                                                                    dim=0).cpu().numpy())

            if jj == args.B - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % args.B == 0 and jj != args.B - 1:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        meas = torch.from_numpy(meas).cuda().float()
        pic_gt = torch.from_numpy(pic_gt).cuda().float()

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)

        out_save1 = torch.zeros([meas.shape[0], args.B, 3, args.size[0], args.size[1]]).cuda()
        with torch.no_grad():

            psnr_1 = 0
            for ii in range(meas.shape[0]):

                model.mask = mask
                out_pic1 = model(meas_re[ii:ii + 1, ::], args)
                out_pic1 = out_pic1.reshape(1, 3, args.B, args.size[0], args.size[1]).permute(0, 2, 1, 3, 4)
                out_save1[ii, :, :, :, :] = out_pic1[0, :, :, :, :]

                for jj in range(args.B):
                    out_pic_forward = out_save1[ii, jj, :, :, :]
                    gt_t = pic_gt[ii, jj, :, :, :]
                    mse_forward = loss(out_pic_forward * 255, gt_t * 255)
                    mse_forward = mse_forward.data
                    psnr_1 += 10 * torch.log10(255 * 255 / mse_forward)

            psnr_1 = psnr_1 / (meas.shape[0] * args.B)
            psnr_cnn[i] = psnr_1

            a = test_list[i]
            name1 = result_path + '/RevSCInet_' + a[0:len(a) - 4] + '{}_{:.4f}'.format(epoch, psnr_1) + '.mat'
            out_save1 = out_save1.cpu()
            scio.savemat(name1, {'pic': out_save1.numpy()}, do_compression=True)
    print("RevSCInet result: {:.4f}".format(torch.mean(psnr_cnn)))


if __name__ == '__main__':
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if args.last_train != 0:
        rev_net = re_3dcnn1(18).cuda()
        rev_net.mask = mask
        rev_net.load_state_dict(torch.load('./model/' + args.model_save_filename + "/RevSCInet_model_epoch_{}.pth".format(args.last_train)))

        rev_net = rev_net.module if hasattr(rev_net, "module") else rev_net
    test(test_path1, args.last_train, result_path, rev_net.eval(), args)
