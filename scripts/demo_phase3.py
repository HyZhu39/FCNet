import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

sys.path.append('.')

from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args
from torchvision.transforms import functional
from torchvision.utils import save_image, make_grid

RESIZE_SIZE = 256

transform = transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.ToTensor(),   ])

def vis_train(img_train_batch, idx):
    # saving training results input NCHW
    img_train_batch = torch.cat(img_train_batch, dim=3)
    save_path = os.path.join(args.save_folder, idx+'.png')
    vis_image = make_grid((img_train_batch), 1)
    save_image(vis_image, save_path)  # , normalize=True)

def vis_train_single(img_train_batch, idx):
    # saving training results
    save_path = args.save_folder + '/single'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, str(idx)+'.png')
    print(save_path)
    #vis_image = de_norm(img_train_batch)
    vis_image = img_train_batch
    save_image(vis_image, save_path)  # , normalize=True)

def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Inference(config, args, args.load_path)

    n_imgname = sorted(os.listdir(args.source_dir))

    for i, (imga_name) in enumerate((n_imgname)):
        imgA = Image.open(os.path.join(args.source_dir, imga_name)).convert('RGB')

        imgA = transforms.Resize(RESIZE_SIZE)(imgA)
        
        result_1 = inference.transfer_phase3(imgA).to('cpu')
        
        imgA = transform(imgA).unsqueeze(0)
        
        basename = imga_name.split('/')[-1].split('.')[0]
        vis_train_single(result_1, basename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='phase_3')
    parser.add_argument("--train_phase", type=str, default='phase_3')
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")

    parser.add_argument("--load_path", type=str, help="folder to load model",
                        default='')

    parser.add_argument("--source-dir", type=str,
                        default="")

    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    config = get_config()
    main(config, args)
