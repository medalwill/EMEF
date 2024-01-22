import torchvision.transforms as transforms
from util.MEF_SSIM import mef_ssim
from .base_model import BaseModel
from . import networks
import torch


def rescale(x):
    return (x + 1) / 2 * 255


class OptimModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_512', dataset_mode='HDR', netD='n_layers', n_layers_D=6)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_MEFSSIM', 'tmp']
        self.visual_names = ['oe', 'ue', 'fake_B', 'oe_gray', 'ue_gray', 'fake_B_gray']
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc * 2, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterion = mef_ssim
            self.set_requires_grad(self.netG, False)

    def set_input(self, data):
        self.oe = data["oe"].to(self.device)
        self.ue = data["ue"].to(self.device)
        self.cls = torch.nn.Parameter(torch.zeros(1, 4), requires_grad=True)
        self.optimizer = torch.optim.Adam([self.cls], lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.image_paths = data['image_name']

    def forward(self):
        input_data = torch.cat([self.oe, self.ue], 1)
        d8, d7, d6, d5, d4, e0, e1, e2, e3 = self.netG(input_data, self.cls)
        self.fake_B = d8

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        transform = transforms.Grayscale()
        self.fake_B_gray = transform(rescale(self.fake_B))
        self.oe_gray = transform(rescale(self.oe))
        self.ue_gray = transform(rescale(self.ue))
        self.img_seq = torch.cat([self.oe_gray, self.ue_gray], 1)
        self.loss_G_MEFSSIM = 1 - self.criterion(self.img_seq, self.fake_B_gray)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_MEFSSIM
        self.loss_tmp = self.loss_G_MEFSSIM
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        self.optimizer.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer.step()             # udpate G's weights
        self.cls.data = torch.clamp(self.cls.data, min=0, max=1)