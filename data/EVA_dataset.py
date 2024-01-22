import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from PIL import Image
import os.path


class EVADataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_images = os.path.join(opt.dataroot, "fake")
        self.oe = os.path.join(opt.dataroot, "oe")
        self.ue = os.path.join(opt.dataroot, "ue")
        self.image_names = [x.replace(".jpg", "").replace(".png", "") for x in os.listdir(self.dir_images)]
        self.isTrain = opt.isTrain

        self.transforms = transforms.Compose([
            transforms.Resize((opt.texture_size, opt.texture_size), Image.BILINEAR),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize((0.5,), (0.5,))

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.dir_images, image_name + ".png")

        image = self.transforms(Image.open(image_path))
        oe_path = os.path.join(self.oe, image_name+'.png')
        ue_path = os.path.join(self.ue, image_name + '.png')
        oe = self.transforms(Image.open(oe_path))
        ue = self.transforms(Image.open(ue_path))

        normalized_image = self.normalize(image)
        normalized_oe = self.normalize(oe)
        normalized_ue = self.normalize(ue)

        return {"gt": normalized_image, "oe": normalized_oe, "ue": normalized_ue, "image_name": image_name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_names)


