import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _mef_ssim(imgSeq, refImg, window, window_size):
    (_, imgSeq_channel, _, _) = imgSeq.size()
    (_, refImg_channel, _, _) = refImg.size()
    C2 = (0.03 * 255)**2
    sWindow = torch.ones((imgSeq_channel,1,window_size,window_size)) / window_size**2
    if refImg.is_cuda:
        sWindow = sWindow.cuda(refImg.get_device())
    mu_x = F.conv2d(imgSeq, sWindow, padding = window_size//2, groups = imgSeq_channel)

    mfilter = torch.ones((imgSeq_channel,1,window_size,window_size))
    x_hat = imgSeq - mu_x
    tmp = torch.pow(x_hat,2)
    if refImg.is_cuda:
        mfilter = mfilter.cuda(refImg.get_device())
        tmp = tmp.cuda(refImg.get_device())
    x_hat_norm = torch.sqrt(F.conv2d(tmp, mfilter, padding = window_size//2, groups = imgSeq_channel))+ 0.001
    c_hat = torch.max(x_hat_norm,dim=1)[0]

    mfilter2 = torch.ones((1,1,window_size,window_size))
    tmp1 = torch.pow(torch.sum(x_hat,1,keepdim=True),2)
    if refImg.is_cuda:
        mfilter2 = mfilter2.cuda(refImg.get_device())
        tmp1 = tmp1.cuda(refImg.get_device())
    R = (torch.sqrt(F.conv2d(tmp1, mfilter2, padding = window_size//2, groups = 1)) +  np.spacing(1)+  np.spacing(1)) \
        / (torch.sum(x_hat_norm,1,keepdim=True) +  np.spacing(1))

    R[R > 1] = 1 -  np.spacing(1)
    R[R < 0] = 0 +  np.spacing(1)

    p = torch.tan(R*np.pi/2)
    p[p>10] = 10

    s = x_hat / x_hat_norm

    s_hat_one = torch.sum((torch.pow(x_hat_norm,p)+np.spacing(1))*s,1,keepdim=True)/torch.sum((torch.pow(x_hat_norm,p)+np.spacing(1)),1,keepdim=True)
    s_hat_two = s_hat_one / torch.sqrt(F.conv2d(torch.pow(s_hat_one,2), mfilter2, padding = window_size//2, groups = refImg_channel))

    x_hat_two = c_hat*s_hat_two

    mu_x_hat_two = F.conv2d(x_hat_two, window, padding = window_size//2, groups = refImg_channel)
    mu_y = F.conv2d(refImg, window, padding = window_size//2, groups = refImg_channel)

    mu_x_hat_two_sq = torch.pow(mu_x_hat_two,2)
    mu_y_sq = torch.pow(mu_y,2)
    mu_x_hat_two_mu_y = mu_x_hat_two * mu_y
    sigma_x_hat_two_sq = F.conv2d(x_hat_two*x_hat_two, window, padding = window_size//2, groups = refImg_channel) - mu_x_hat_two_sq
    sigma_y_sq = F.conv2d(refImg*refImg, window, padding = window_size//2, groups = refImg_channel) - mu_y_sq
    sigmaxy = F.conv2d(x_hat_two*refImg, window, padding = window_size//2, groups = refImg_channel) - mu_x_hat_two_mu_y

    mef_ssim_map = (2*sigmaxy + C2)/(sigma_x_hat_two_sq + sigma_y_sq + C2)

    return mef_ssim_map.mean()


class MEF_SSIM(torch.nn.Module):
    def __init__(self, window_size = 11):
        super(MEF_SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img_seq, refImg):
        if img_seq.is_cuda:
            window = window.cuda(img_seq.get_device())
        window = window.type_as(img_seq)
        
        return _mef_ssim(img_seq, refImg, self.window, self.window_size)


def mef_ssim(img_seq, refImg, window_size = 11):
    (_, channel, _, _) = refImg.size()
    window = create_window(window_size, channel)
    
    if img_seq.is_cuda:
        window = window.cuda(img_seq.get_device())
    window = window.type_as(img_seq)
    
    return _mef_ssim(img_seq, refImg, window, window_size)
