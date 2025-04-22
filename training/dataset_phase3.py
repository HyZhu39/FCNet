import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from training.config import get_config
from training.preprocess import PreProcess
import random
from random import sample
import numpy as np
import face_util as futils
import torchvision
from torch.nn import functional as F
import math
import torch.nn as nn
import albumentations

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm')


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    # code from BasicSR codebase
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


class MakeupDataset(Dataset):
    def __init__(self, config=None):
        super(MakeupDataset, self).__init__()
        if config is None:
            config = get_config()
        self.gt_path = config.DATA.GT_PATH
        self.rate = config.DATA.PART_JITTER_RATE
        self.mask_path = config.DATA.MASK_PATH

        scaner = scandir(dir_path=self.gt_path, suffix=IMG_EXTENSIONS, recursive=True, full_path=True)
        paths = []
        paths += sorted(scaner)
        self.makeup_names = paths

        self.flip_rate = 0.5
        self.aug_rate = 0.75

        self.preprocessor = PreProcess(config, need_parser=False)
        self.img_size = config.DATA.IMG_SIZE
        self.domain_num = config.MODEL.DOMAINS

        self.transform = transforms.Compose([
            transforms.Resize(config.DATA.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # 0,1

        self.transform_grey = transforms.Compose([
            transforms.Resize(config.DATA.IMG_SIZE),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # 0,1

        self.transform_jitter = transforms.Compose([
            transforms.Resize(config.DATA.IMG_SIZE),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])  # 0,1

    def load_from_file(self, img_name):
        image = Image.open(os.path.join(self.root, 'images', img_name)).convert('RGB')
        base_name = os.path.splitext(img_name)[0]
        mask = self.preprocessor.load_mask(os.path.join(self.root, 'segs', f'{base_name}.png'))
        lms = self.preprocessor.load_lms(os.path.join(self.root, 'lms', f'{base_name}.npy'))
        return self.preprocessor.process(image, mask, lms)

    def load_from_file_gt(self, img_name):  # 'non-makeup/19946'
        image = Image.open(img_name).convert('RGB')
        if img_name.split('.')[-1] == 'jpg':
            mask_path = img_name.replace(self.gt_path, self.mask_path).replace('.jpg', '.png')
        else:
            mask_path = img_name.replace(self.gt_path, self.mask_path)

        mask = self.preprocessor.load_mask(mask_path)
        return self.preprocessor.process_train(image, mask)

    def __len__(self):
        return len(self.makeup_names)

    def __getitem__(self, index):
        idx_r = torch.randint(0, len(self.makeup_names), (1,)).item()

        name_r = self.makeup_names[idx_r]

        reference = self.load_from_file_gt(name_r)

        mask = reference[1]  # mask torch.Size([5, 256, 256])
        mask = remake_masks(torch.unsqueeze(mask, 0))[0]  # mask of input grey image

        reference_gt_ = Image.open(name_r).convert('RGB')

        reference_gt = self.transform(reference_gt_)
        reference_grey = self.transform_grey(reference_gt_)
        # flip
        if np.random.rand() > self.flip_rate:
            grey_ = reference_grey
            mask_ = mask
            gt_ = reference_gt
        else:
            grey_ = torch.flip(reference_grey, dims=[-1])
            mask_ = torch.flip(mask, dims=[-1])
            gt_ = torch.flip(reference_gt, dims=[-1])

        if (np.random.rand() > self.aug_rate):
            grey__ = grey_
            mask__ = mask_
            gt__ = gt_
        else:
            angle = np.random.randint(-30, 30, size=1)[0]  # -30,30
            ratio = np.around(random.uniform(1.3, 1.65), decimals=2)  # 1.3 - 1.65
            grey__, mask__, gt__ = affine_augmention_3(grey_, mask_, gt_, angle, ratio)

        return reference_gt, reference_grey, mask, grey__, mask__, gt__

    def get_location(self, img, ratio=0.85):  #

        up_ratio = 0.6 / ratio
        down_ratio = 0.2 / 0.85
        width_ratio = 0.2 / 0.85

        face = futils.detect(img)
        if not face:
            return None, None
        location1, location2 = futils.crop_face_alter_location(img, face[0], up_ratio, down_ratio, width_ratio)
        return location1, location2

    def crop_with2location(self, img, location1, location2):
        img = torchvision.transforms.functional.crop(img, location1[1], location1[0], location1[3] - location1[1],
                                                     location1[2] - location1[
                                                         0])  # top: int, left: int, height: int, width: int
        img = torchvision.transforms.functional.crop(img, location2[1], location2[0], location2[3] - location2[1],
                                                     location2[2] - location2[0])
        return img

    def convert_part_color(self, img_path, mask, rate):
        '''
        img:  input image path  #torch.Size([3, 256, 256])
        mask: torch.Size([5, 256, 256])
        [mask_lip, mask_face, mask_eye_left, mask_eye_right, mask_hair]
        rate: random rate, 0-1
        '''
        random_rate = random.uniform(0, 1)
        img = Image.open(img_path).convert('RGB')  # torch.Size([3, 256, 256])

        if random_rate <= rate:  # jitter
            img_alter = []

            C_part, _, _ = mask.shape
            for ids in range(C_part):
                img_alter.append(self.transform_jitter(img) * mask[ids:ids + 1])

            img_alter = torch.stack(img_alter, dim=0)  # torch.Size([5, 3, 256, 256])
            img_alter = torch.sum(img_alter, dim=0)  # torch.Size([3, 256, 256])

            img_alter_grey = self.transform_grey(self.tensor_2_PIL(img_alter))

            return img_alter, img_alter_grey
        else:
            jitter = self.transform_jitter(img)
            jitter_grey = self.tensor_2_PIL(jitter)
            return jitter, self.transform_grey(jitter_grey)

    def merge_part_color(self, jittered_imgs, mask):
        '''
        jittered_imgs:  list [tensor([3,256,256])]
        mask: torch.Size([5, 256, 256])
        '''
        img_alter = []
        C_part, _, _ = mask.shape
        for ids in range(C_part):
            img_alter.append(jittered_imgs[ids] * mask[ids:ids + 1])
        img_alter = torch.stack(img_alter, dim=0)  #
        img_alter = torch.sum(img_alter, dim=0)  #
        return img_alter

    def tensor_2_PIL(self, tensor_img):  # CHW -1,1
        tensor = tensor_img
        tensor = tensor.cpu().clone()
        tensor = self.de_norm(tensor)
        tensor = tensor.permute(1, 2, 0)
        image = tensor.numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        return image

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)


def remake_masks(masks):
    '''
    masks:     N domain_num HW
    # lip skin left_eye right_eye hair
    '''
    masks[:, 2:3] = masks[:, 2:4].sum(dim=1, keepdim=True)
    masks[:, 2:3][masks[:, 2:3] > 1] = 1  # attention this!!!
    masks = torch.concat((masks[:, 0:3], masks[:, 4:]), dim=1)
    masks[masks > 1] = 1  # ensure no overlap

    mask_background = masks.sum(dim=1, keepdim=True)  # torch.zeros_like(masks[:, 0:1, :, :]) # N 1 256 256
    mask_background[mask_background != 0] = -1
    mask_background += 1

    masks = torch.cat([masks, mask_background], dim=1)
    return masks


def affine_augmention(img, mask, angle, ratio):
    '''
    Args:
        img:   torch tensor CHW
        mask:  torch tensor C'HW
        angle: -30,30
        ratio: 1.3 - 1.65
    Returns:
        aug:        torch tensor CHW
        aug_mask:   torch tensor C'HW
    '''
    angle = angle * math.pi / 180
    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), 0],
        [math.sin(angle), math.cos(angle), 0]
    ], dtype=torch.float)
    center = (img.size()[-2], img.size()[-1])

    C, H, W = img.shape
    aug_size = torch.Size([1, C, int(H * ratio), int(W * ratio)])
    grid = F.affine_grid(theta.unsqueeze(0), aug_size, align_corners=True)
    img = F.grid_sample(img.unsqueeze(0), grid, align_corners=True)
    img = transforms.CenterCrop(center)(img)
    img = img[0]

    gridm = F.affine_grid(theta.unsqueeze(0), aug_size, align_corners=True)
    mask = F.grid_sample(mask.unsqueeze(0), gridm, align_corners=True)
    mask = transforms.CenterCrop(center)(mask)
    mask = mask[0]

    return img, mask


def affine_augmention_3(img, mask, gt, angle, ratio):
    '''
    Args:
        img:   torch tensor CHW
        gt:    torch tensor CHW
        mask:  torch tensor C'HW
        angle: -30,30
        ratio: 1.3 - 1.65
    Returns:
        aug:        torch tensor CHW
        aug_mask:   torch tensor C'HW
        aug_gt:     torch tensor CHW
    '''
    angle = angle * math.pi / 180
    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), 0],
        [math.sin(angle), math.cos(angle), 0]
    ], dtype=torch.float)
    center = (img.size()[-2], img.size()[-1])

    C, H, W = img.shape
    aug_size = torch.Size([1, C, int(H * ratio), int(W * ratio)])
    grid = F.affine_grid(theta.unsqueeze(0), aug_size, align_corners=True)
    img = F.grid_sample(img.unsqueeze(0), grid, align_corners=True)
    img = transforms.CenterCrop(center)(img)
    img = img[0]

    gt = F.grid_sample(gt.unsqueeze(0), grid, align_corners=True)
    gt = transforms.CenterCrop(center)(gt)
    gt = gt[0]

    gridm = F.affine_grid(theta.unsqueeze(0), aug_size, align_corners=True)
    mask = F.grid_sample(mask.unsqueeze(0), gridm, align_corners=True)
    mask = transforms.CenterCrop(center)(mask)
    mask = mask[0]

    return img, mask, gt


def get_loader(config):
    dataset = MakeupDataset(config)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.DATA.BATCH_SIZE,
                            num_workers=config.DATA.NUM_WORKERS)
    return dataloader


if __name__ == "__main__":
    dataset = MakeupDataset()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=16)
    for e in range(10):
        for i, (point_s, point_r) in enumerate(dataloader):
            pass