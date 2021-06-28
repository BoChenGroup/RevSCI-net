from my_tools import *
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler



class re_3dcnn1(nn.Module):

    def __init__(self, units):
        super(re_3dcnn1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1),
        )
        self.fuse_r = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1)
        )

        self.layers = nn.ModuleList()
        for i in range(units):
            self.layers.append(rev_3d_part(32))

    def forward(self, meas_re, args):
        size1 = [args.size[0], args.size[1]]
        batch_size = meas_re.shape[0]
        mask = self.mask.to(meas_re.device)
        maskt = mask.expand([batch_size, args.B, size1[0], size1[1]])

        # for it in range(4):
        #     if it == 0:
        #         data1 = meas_re[:, :, 0:args.size[0]:2, 0:args.size[1]:2].mul(
        #             maskt[:, :, 0:args.size[0]:2, 0:args.size[1]:2])
        #         data1 = torch.unsqueeze(data1, 1)
        #     elif it == 1:
        #         data1 = torch.cat([data1, torch.unsqueeze(meas_re[:, :, 1:args.size[0]:2, 0:args.size[1]:2].mul(
        #             maskt[:, :, 1:args.size[0]:2, 0:args.size[1]:2]), 1)], dim=1)
        #     elif it == 2:
        #         data1 = torch.cat([data1, torch.unsqueeze(meas_re[:, :, 0:args.size[0]:2, 1:args.size[1]:2].mul(
        #             maskt[:, :, 0:args.size[0]:2, 1:args.size[1]:2]), 1)], dim=1)
        #     elif it == 3:
        #         data1 = torch.cat([data1, torch.unsqueeze(meas_re[:, :, 1:args.size[0]:2, 1:args.size[1]:2].mul(
        #             maskt[:, :, 0:args.size[0]:2, 1:args.size[1]:2]), 1)], dim=1)

        data1 = torch.cat([torch.unsqueeze(maskt, dim=1),
                           torch.unsqueeze(meas_re.expand([batch_size, args.B, size1[0], size1[1]]), dim=1)], dim=1)

        out = self.conv1(data1)

        for layer in self.layers:
            out = layer(out)

        out = self.conv2(out)

        return out

    def for_backward(self, meas_re, gt, loss, opt, args):
        batch_size = meas_re.shape[0]
        mask = self.mask.to(meas_re.device)
        maskt = mask.expand([batch_size, args.B, args.size[0], args.size[1]])

        # for it in range(4):
        #     if it == 0:
        #         data1 = meas_re[:, :, 0:args.size[0]:2, 0:args.size[1]:2].mul(
        #             maskt[:, :, 0:args.size[0]:2, 0:args.size[1]:2])
        #         data1 = torch.unsqueeze(data1, 1)
        #     elif it == 1:
        #         data1 = torch.cat([data1, torch.unsqueeze(meas_re[:, :, 1:args.size[0]:2, 0:args.size[1]:2].mul(
        #             maskt[:, :, 1:args.size[0]:2, 0:args.size[1]:2]), 1)], dim=1)
        #     elif it == 2:
        #         data1 = torch.cat([data1, torch.unsqueeze(meas_re[:, :, 0:args.size[0]:2, 1:args.size[1]:2].mul(
        #             maskt[:, :, 0:args.size[0]:2, 1:args.size[1]:2]), 1)], dim=1)
        #     elif it == 3:
        #         data1 = torch.cat([data1, torch.unsqueeze(meas_re[:, :, 1:args.size[0]:2, 1:args.size[1]:2].mul(
        #             maskt[:, :, 0:args.size[0]:2, 1:args.size[1]:2]), 1)], dim=1)

        data1 = torch.cat([torch.unsqueeze(maskt, dim=1),
                           torch.unsqueeze(meas_re.expand([batch_size, args.B, args.size[0], args.size[1]]), dim=1)], dim=1)

        with torch.no_grad():
            out1 = self.conv1(data1)
            out2 = out1
            for layer in self.layers:
                out2 = layer(out2)
        out3 = out2.requires_grad_()
        out4 = self.conv2(out3)

        loss1 = loss(out4, gt)
        loss1.backward()
        current_state_grad = out3.grad

        out_current = out3
        for layer in reversed(self.layers):
            with torch.no_grad():
                out_pre = layer.reverse(out_current)
            out_pre.requires_grad_()
            out_cur = layer(out_pre)
            torch.autograd.backward(out_cur, grad_tensors=current_state_grad)
            current_state_grad = out_pre.grad
            out_current = out_pre

        out1 = self.conv1(data1)
        out1.requires_grad_()
        torch.autograd.backward(out1, grad_tensors=current_state_grad)
        if opt != 0:
            opt.step()

        return out4, loss1
