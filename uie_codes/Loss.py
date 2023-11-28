import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import exp
import numpy as np


class MyMSELoss(nn.Module):
    def __init__(self):
        super(MyMSELoss, self).__init__()

    def forward(self, xs, ys, trans):
        loss = torch.mean((xs - ys) ** 2)
        return loss


class MyMAELoss(nn.Module):
    def __init__(self):
        super(MyMAELoss, self).__init__()

    def forward(self, xs, ys, trans=None):
        loss = torch.mean(torch.abs(xs - ys))
        return loss
    
# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 
 
# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
 
 
# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
 
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):

    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        # print("sim",sim)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1]) #返回所有元素的乘积
    return output

# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=7, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
 
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
 
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
 
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
 
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average, val_range=self.val_range)
    
class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=7, size_average=True, val_range=None):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.val_range = val_range

    def forward(self, img1, img2):

        # TODO: store window between calls if possible,
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, val_range=self.val_range, normalize=True)
   


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def normal(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized 

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).cuda().eval()
        vgg_model = vgg_model.features
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '31': "relu5_2"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return F.mse_loss(input, target)
    
    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return F.mse_loss(input_mean, target_mean) + F.mse_loss(input_std, target_std)
    
    def forward(self, pred_im, gt, content_index=[3, 4]):
        # style_index = [0, 1, 2, 3, 4]
        content_loss_list = []
        # style_loss_list = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)

        for i in content_index:
            content_loss_list.append(self.calc_content_loss(normal(pred_im_features[i]), normal(gt_features[i])))

        # for i in style_index:
        #     style_loss_list.append(self.calc_style_loss(pred_im_features[i], gt_features[i]))

        content_loss = sum(content_loss_list)/len(content_loss_list)
        # style_loss = sum(style_loss_list)/len(style_loss_list)

        return content_loss



# l = PerceptualLoss()
# x = torch.randn(1, 3, 256, 256).cuda()
# y = torch.randn(1, 3, 256, 256).cuda()
# print(l(x, y))


class ColorLoss(torch.nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, img):
        channel_mean = torch.mean(img, dim=(2, 3), keepdim=True)
        mr, mg, mb = torch.split(channel_mean, 1, dim=1)
        # channel_var = torch.var(img, dim=(2, 3))

        Drg = torch.pow(mr-mg, 2)
        Drb = torch.pow(mr-mb, 2)
        Dgb = torch.pow(mb-mg, 2)
        color_constant_loss = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        # RG_diff_var = F.mse_loss(channel_var[:, 0], channel_var[:, 1])
        # GB_diff_var = F.mse_loss(channel_var[:, 1], channel_var[:, 2])
        # RB_diff_var = F.mse_loss(channel_var[:, 0], channel_var[:, 2])
        # color_dispersion_loss = RG_diff_var + GB_diff_var + RB_diff_var

        return torch.mean(color_constant_loss) #+ color_dispersion_loss
    
class IlluminationLoss(nn.Module):
    def __init__(self):
        super(IlluminationLoss, self).__init__()

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size


