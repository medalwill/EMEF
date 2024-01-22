from torch_utils import conv2d_resample
from torch_utils import upfirdn2d
from torch_utils import bias_act

from torch.optim import lr_scheduler
from torch.nn import init
import torch.nn as nn
import torch

import numpy as np
import functools


###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x
    
    
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None

    if netG == 'unet_512':
        net = Unet512(input_nc, output_nc, ngf)
    elif netG == 'unet_256':
        net = Unet256(input_nc, output_nc, ngf)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


def patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def load_network(net, load_path, device):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    print('loading the model from %s' % load_path)
    state_dict = torch.load(load_path, map_location=str(device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    net.load_state_dict(state_dict)
    return net


##############################################################################
# Classes
##############################################################################

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, norm, use_dropout=True):
        super(Decoder, self).__init__()

        use_bias = False
        self.relu = nn.ReLU(True)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.norm = norm(out_channels)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x2):
        x1 = self.relu(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.up(x1)
        x1 = self.norm(x1)
        if self.use_dropout:
            x1 = self.dropout(x1)
        return x1


class DecoderFinal(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderFinal, self).__init__()
        self.relu = nn.ReLU(True)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x1, x2):
        x1 = self.relu(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.up(x1)
        x1 = self.tanh(x1)
        return x1


class Unet512(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm=nn.InstanceNorm2d):
        super(Unet512, self).__init__()

        use_bias = False
        w_dim = 512

        self.encoder0 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1,
                                  bias=use_bias)  # (6, 512, 512) => (64, 256, 256)
        self.encoder1 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 2))  # (64, 256, 256) => (128, 128, 128)
        self.encoder2 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 4))  # (128, 128, 128) => (256, 64, 64)
        self.encoder3 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 8))  # (256, 64, 64) => (512, 32, 32)
        self.encoder4 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 8))  # (512, 32, 32) => (512, 16, 16)
        self.encoder5 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 8))  # (512, 16, 16) => (512, 8, 8)
        self.encoder6 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 8))  # (512, 8, 8) => (512, 4, 4)
        self.encoder7 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 8))  # (512, 4, 4) => (512, 2, 2)
        self.encoder8 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1,
                                                bias=use_bias))  # (512, 2, 2) => (512, 1, 1)

        self.decoder0 = nn.Sequential(nn.ReLU(True),
                                      nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1,
                                                         bias=use_bias),
                                      norm(ngf * 8))  # (512, 1, 1) => (512, 2, 2)
        self.block0 = SynthesisLayer(512, 512, w_dim=w_dim)
        self.decoder1 = Decoder(ngf * 8 + ngf * 8, ngf * 8, norm)  # (512+512, 2, 2) => (512, 4, 4)
        self.block1 = SynthesisLayer(512, 512, w_dim=w_dim)
        self.decoder2 = Decoder(ngf * 8 + ngf * 8, ngf * 8, norm)  # (512+512, 4, 4) => (512, 8, 8)
        self.block2 = SynthesisLayer(512, 512, w_dim=w_dim)
        self.decoder3 = Decoder(ngf * 8 + ngf * 8, ngf * 8, norm)  # (512+512, 8, 8) => (512, 16, 16)
        self.block3 = SynthesisLayer(512, 512, w_dim=w_dim)
        self.decoder4 = Decoder(ngf * 8 + ngf * 8, ngf * 8, norm)  # (512+512, 16, 16) => (512, 32, 32)
        self.block4 = SynthesisLayer(512, 512, w_dim=w_dim)
        self.decoder5 = Decoder(ngf * 8 + ngf * 8, ngf * 4, norm, False)  # (512+512, 32, 32) => (256, 64, 64)
        self.block5 = SynthesisLayer(256, 256, w_dim=w_dim)
        self.decoder6 = Decoder(ngf * 4 + ngf * 4, ngf * 2, norm, False)  # (256+256, 64, 64) => (128, 128, 128)
        self.block6 = SynthesisLayer(128, 128, w_dim=w_dim)
        self.decoder7 = Decoder(ngf * 2 + ngf * 2, ngf, norm, False)  # (128+128, 128, 128) => (64, 256, 256)
        self.block7 = SynthesisLayer(64, 64, w_dim=w_dim)
        self.decoder8 = DecoderFinal(ngf + ngf, output_nc)  # (64+64, 256, 256) => (3, 512, 512)

        self.mapping = MappingNetwork(z_dim=4, c_dim=0, w_dim=w_dim, num_ws=8)

    def forward(self, data,  latents):
        ws = self.mapping(latents, None)
        w_iter = iter(ws.unbind(dim=1))

        e0 = self.encoder0(data)  # (6, 512, 512) => (64, 256, 256)
        e1 = self.encoder1(e0)  # (64, 256, 256) => (128, 128, 128)
        e2 = self.encoder2(e1)  # (128, 128, 128) => (256, 64, 64)
        e3 = self.encoder3(e2)  # (256, 64, 64) => (512, 32, 32)
        e4 = self.encoder4(e3)  # (512, 32, 32) => (512, 16, 16)
        e5 = self.encoder5(e4)  # (512, 16, 16) => (512, 8, 8)
        e6 = self.encoder6(e5)  # (512, 8, 8) => (512, 4, 4)
        e7 = self.encoder7(e6)  # (512, 4, 4) => (512, 2, 2)
        e8 = self.encoder8(e7)  # (512, 2, 2) => (512, 1, 1)

        d0 = self.decoder0(e8)  # (512, 1, 1) => (512, 2, 2)
        d0 = self.block0(d0, next(w_iter), True)

        d1 = self.decoder1(d0, e7)  # (1024, 2, 2) => (512, 4, 4)
        d1 = self.block1(d1, next(w_iter), True)
        d2 = self.decoder2(d1, e6)  # (1024, 4, 4) => (512, 8, 8)
        d2 = self.block2(d2, next(w_iter), True)
        d3 = self.decoder3(d2, e5)  # (1024, 8, 8) => (512, 16, 16)
        d3 = self.block3(d3, next(w_iter), True)
        d4 = self.decoder4(d3, e4)  # (1024, 16, 16) => (512, 32, 32)
        d4 = self.block4(d4, next(w_iter), True)
        d5 = self.decoder5(d4, e3)  # (1024, 32, 32) => (256, 64, 64)
        d5 = self.block5(d5, next(w_iter), True)
        d6 = self.decoder6(d5, e2)  # (512, 64, 64) => (128, 128, 128)
        d6 = self.block6(d6, next(w_iter), True)
        d7 = self.decoder7(d6, e1)  # (256, 128, 128) => (64, 256, 256)
        d7 = self.block7(d7, next(w_iter), True)
        d8 = self.decoder8(d7, e0)  # (128, 256, 256) => (3, 512, 512)

        return d8, d7, d6, d5, d4, e0, e1, e2, e3


class Unet256(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm=nn.InstanceNorm2d):
        super(Unet256, self).__init__()

        use_bias = False
        w_dim = 512

        self.encoder0 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1,
                                  bias=use_bias)  # (6, 256, 256) => (64, 128, 128)
        self.encoder1 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 2))  # (64, 128, 128) => (128, 64, 64)
        self.encoder2 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 4))  # (128, 64, 64) => (256, 32, 32)
        self.encoder3 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 8))  # (256, 32, 32) => (512, 16, 16)
        self.encoder4 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 8))  # (512, 16, 16) => (512, 8, 8)
        self.encoder5 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 8))  # (512, 8, 8) => (512, 4, 4)
        self.encoder6 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm(ngf * 8))  # (512, 4, 4) => (512, 2, 2)
        self.encoder7 = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias))
        # (512, 2, 2) => (512, 1, 1)

        self.decoder0 = nn.Sequential(nn.ReLU(True),
                                      nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1,
                                                         bias=use_bias),
                                      norm(ngf * 8))  # (512, 1, 1) => (512, 2, 2)
        self.block0 = SynthesisLayer(512, 512, w_dim=w_dim)
        self.decoder1 = Decoder(ngf * 8 + ngf * 8, ngf * 8, norm)  # (512+512, 2, 2) => (512, 4, 4)
        self.block1 = SynthesisLayer(512, 512, w_dim=w_dim)
        self.decoder2 = Decoder(ngf * 8 + ngf * 8, ngf * 8, norm)  # (512+512, 4, 4) => (512, 8, 8)
        self.block2 = SynthesisLayer(512, 512, w_dim=w_dim)
        self.decoder3 = Decoder(ngf * 8 + ngf * 8, ngf * 8, norm)  # (512+512, 8, 8) => (512, 16, 16)
        self.block3 = SynthesisLayer(512, 512, w_dim=w_dim)
        self.decoder4 = Decoder(ngf * 8 + ngf * 8, ngf * 4, norm, False)  # (512+512, 16, 16) => (256, 32, 32)
        self.block4 = SynthesisLayer(256, 256, w_dim=w_dim)
        self.decoder5 = Decoder(ngf * 4 + ngf * 4, ngf * 2, norm, False)  # (256+256, 32, 32) => (128, 64, 64)
        self.block5 = SynthesisLayer(128, 128, w_dim=w_dim)
        self.decoder6 = Decoder(ngf * 2 + ngf * 2, ngf, norm, False)  # (128+128, 64, 64) => (64, 128, 128)
        self.block6 = SynthesisLayer(64, 64, w_dim=w_dim)
        self.decoder7 = DecoderFinal(ngf + ngf, output_nc)  # (64+64, 128, 128) => (3, 256, 256)

        self.mapping = MappingNetwork(z_dim=4, c_dim=0, w_dim=w_dim, num_ws=7)

    def forward(self, data,  latents):
        ws = self.mapping(latents, None)
        w_iter = iter(ws.unbind(dim=1))

        e0 = self.encoder0(data)  # (6, 256, 256) => (64, 128, 128)
        e1 = self.encoder1(e0)  # (64, 128, 128) => (128, 64, 64)
        e2 = self.encoder2(e1)  # (128, 64, 64) => (256, 32, 32)
        e3 = self.encoder3(e2)  # (256, 32, 32) => (512, 16, 16)
        e4 = self.encoder4(e3)  # (512, 16, 16) => (512, 8, 8)
        e5 = self.encoder5(e4)  # (512, 8, 8) => (512, 4, 4)
        e6 = self.encoder6(e5)  # (512, 4, 4) => (512, 2, 2)
        e7 = self.encoder7(e6)  # (512, 2, 2) => (512, 1, 1)

        d0 = self.decoder0(e7)  # (512, 1, 1) => (512, 2, 2)
        d0 = self.block0(d0, next(w_iter), True)

        d1 = self.decoder1(d0, e6)  # (1024, 2, 2) => (512, 4, 4)
        d1 = self.block1(d1, next(w_iter), True)
        d2 = self.decoder2(d1, e5)  # (1024, 4, 4) => (512, 8, 8)
        d2 = self.block2(d2, next(w_iter), True)
        d3 = self.decoder3(d2, e4)  # (1024, 8, 8) => (512, 16, 16)
        d3 = self.block3(d3, next(w_iter), True)
        d4 = self.decoder4(d3, e3)  # (1024, 16, 16) => (512, 32, 32)
        d4 = self.block4(d4, next(w_iter), True)
        d5 = self.decoder5(d4, e2)  # (512, 32, 32) => (128, 64, 64)
        d5 = self.block5(d5, next(w_iter), True)
        d6 = self.decoder6(d5, e1)  # (256, 64, 64) => (64, 128, 128)
        d6 = self.block6(d6, next(w_iter), True)
        d7 = self.decoder7(d6, e0)  # (128, 256, 256) => (3, 512, 512)

        return d7, d6, d5, d4, e0, e1, e2


###############################################################################
# Start of StyleGAN2 Implement
# codes from https://github.com/NVlabs/stylegan2-ada-pytorch/
###############################################################################

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        # scale the weight
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                # scale the bias
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x
    

def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1, 2, 3],
                                                                            keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down,
                                            padding=padding, flip_weight=flip_weight)
        if demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        return x

    # Execute as one fused op using grouped convolution.
    batch_size = int(batch_size)
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding,
                                        groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
    ):
        super().__init__()
        self.up = up
        self.activation = activation
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, fused_modconv=True, gain=1):
        styles = self.affine(w)

        flip_weight = (self.up == 1)  # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=None)
        return x


###############################################################################
# End of StyleGAN2 Implement
# codes from https://github.com/NVlabs/stylegan2-ada-pytorch/
###############################################################################

