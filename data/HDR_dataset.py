import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from PIL import Image
import os.path
import torch


class HDRDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.stage = opt.stage
        self.dir_images = os.path.join(opt.dataroot, "gt")
        self.oe = os.path.join(opt.dataroot, "oe")
        self.ue = os.path.join(opt.dataroot, "ue")
        self.image_names = [x.replace(".jpg", "").replace(".png", "") for x in os.listdir(self.dir_images)]
        self.isTrain = opt.isTrain

        self.transforms = transforms.Compose([
            transforms.Resize((opt.texture_size, opt.texture_size), Image.BILINEAR),
            transforms.ToTensor(),
        ])
        self.gray_transforms = transforms.Compose([
            transforms.Resize((opt.texture_size, opt.texture_size), Image.BILINEAR),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize((0.5,), (0.5,))

    def one_hot(self, x, class_count):
        return torch.eye(class_count)[x, :]

    def get_one_hot(self, img_name):
        tmp = int(img_name[-2:])
        class_count = 4
        r = self.one_hot(tmp, class_count)
        return r

    def soft_label(self, cls):
        tmp = []
        for i in cls:
            if i == 0:
                tmp.append(torch.rand(1) * 0.5)
            else:
                tmp.append(1 - torch.rand(1) * 0.5)
        return torch.Tensor(tmp)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.dir_images, image_name + ".png")

        image = self.transforms(Image.open(image_path))
        if self.stage == 1:
            oe_path = os.path.join(self.oe, str(int(image_name[0:3]))+'.png')
            ue_path = os.path.join(self.ue, str(int(image_name[0:3])) + '.png')
        else:
            oe_path = os.path.join(self.oe, image_name + '.png')
            ue_path = os.path.join(self.ue, image_name + '.png')
        oe = self.transforms(Image.open(oe_path))
        ue = self.transforms(Image.open(ue_path))

        normalized_image = self.normalize(image)
        normalized_oe = self.normalize(oe)
        normalized_ue = self.normalize(ue)

        if self.stage == 1:
            cls = self.get_one_hot(image_name)
            cls = self.soft_label(cls)
            return {"gt": normalized_image, "oe": normalized_oe, "ue": normalized_ue, "image_name": image_name,
                    "cls": cls}
        else:
            return {"gt": normalized_image, "oe": normalized_oe, "ue": normalized_ue, "image_name": image_name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_names)
