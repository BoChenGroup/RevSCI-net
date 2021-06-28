
from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from models import re_3dcnn1
from utils import generate_masks, time2file_name, split_masks
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
import argparse
import random
from torch.autograd import Variable
# from thop import profile
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
n_gpu = torch.cuda.device_count()
print('The number of GPU is {}'.format(n_gpu))

data_path = "./largescale_rgb"
test_path1 = "./test/large24"

parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')
parser.add_argument('--last_train', default=0, type=int, help='pretrain model')
parser.add_argument('--model_save_filename', default='', type=str, help='pretrain model save folder name')
parser.add_argument('--max_iter', default=100, type=int, help='max epoch')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--B', default=24, type=int, help='compressive rate')
parser.add_argument('--learning_rate', default=0.0002, type=float)
parser.add_argument('--size', default=[1080, 1920], type=int, help='input image resolution')
parser.add_argument('--mode', default='reverse', type=str, help='training mode: reverse or normal')
parser.add_argument('--save', default=False, type=bool, help='save the test result or not')
args = parser.parse_args()

mask, mask_s = generate_masks(data_path)

dataset = Imgdataset(data_path)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

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
            if args.save == True:
                scio.savemat(name1, {'pic': out_save1.numpy()}, do_compression=True)
    print("RevSCInet result: {:.4f}".format(torch.mean(psnr_cnn)))


def train(epoch, result_path, model, args, rgb2raw):
    epoch_loss = 0
    begin = time.time()

    optimizer_g = optim.Adam([{'params': model.parameters()}], lr=args.learning_rate)


    for iteration, batch in tqdm(enumerate(train_data_loader)):
        gt = Variable(batch)
        gt = gt.cuda().float()
        gt1 = gt * rgb2raw.expand([gt.shape[0], args.B, 3, args.size[0], args.size[1]]).permute(0, 2, 1, 3, 4)
        gt1 = torch.sum(gt1, dim=1)

        model.mask = mask

        maskt = mask.expand([gt.shape[0], args.B, args.size[0], args.size[1]])
        meas = torch.mul(maskt, gt1)
        meas = torch.sum(meas, dim=1)

        meas_re = torch.div(meas, mask_s)
        meas_re = torch.unsqueeze(meas_re, 1)

        optimizer_g.zero_grad()

        if args.mode == 'normal':
            xt1 = model(meas_re, args)
            Loss1 = loss(xt1, gt)
            Loss1.backward()
            optimizer_g.step()
        elif args.mode == 'reverse':
            xt1, Loss1 = model.for_backward(meas_re, gt, loss, optimizer_g, args)


        epoch_loss += Loss1.data


    model = model.module if hasattr(model, "module") else model
    test(test_path1, epoch, result_path, model.eval(), args)
    end = time.time()
    print("===> Epoch {} Complete: Avg. Loss: {:.7f}".format(epoch, epoch_loss / len(train_data_loader)),
          "  time: {:.2f}".format(end - begin))


def checkpoint(epoch, model_path):
    model_out_path = './' + model_path + '/' + "RevSCInet_model_epoch_{}.pth".format(epoch)
    torch.save(rev_net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def main(model, args):
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

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

    for epoch in range(args.last_train + 1, args.last_train + args.max_iter + 1):
        train(epoch, result_path, model, args, rgb2raw)
        if (epoch % 5 == 0) and (epoch < 150):
            args.learning_rate = args.learning_rate * 0.95
            print(args.learning_rate)
        if (epoch % 5 == 0 or epoch > 0):
            model = model.module if hasattr(model, "module") else model
            checkpoint(epoch, model_path)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)


if __name__ == '__main__':
    print(args.mode)
    print(args.learning_rate)

    rev_net = re_3dcnn1(18).cuda()
    rev_net.mask = mask
    if n_gpu > 1:
        rev_net = torch.nn.DataParallel(rev_net)
    if args.last_train != 0:
        rev_net = torch.load(
            './model/' + args.model_save_filename + "/RevSCInet_model_epoch_{}.pth".format(args.last_train))
        rev_net = rev_net.module if hasattr(rev_net, "module") else rev_net

    main(rev_net, args)
