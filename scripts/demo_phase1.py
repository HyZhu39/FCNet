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

batch_size = 1

def remake_masks(masks):
    '''
    masks:     N domain_num HW
    # lip skin left_eye right_eye hair
    '''
    masks[:, 2:3] = masks[:, 2:4].sum(dim=1, keepdim=True)
    masks[:, 2:3][masks[:, 2:3] > 1] = 1                              # attention this!!!
    masks = torch.concat((masks[:, 0:3], masks[:, 4:]), dim=1) 
    masks[masks > 1] = 1                                              # ensure no overlap

    mask_background = masks.sum(dim=1, keepdim=True)
    mask_background[mask_background != 0] = -1
    mask_background += 1

    masks = torch.cat([masks, mask_background], dim=1)
    return masks

transform_gt = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]) # 0,1

def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Inference(config, args, args.load_path)

    n_imgname = sorted(os.listdir(args.source_dir))

    m_imgname = sorted(os.listdir(args.reference_dir))
    
    for i, (imga_name, imgb_name) in enumerate(zip(n_imgname, m_imgname)):
        reference_grey = []
        reference_jitter = []
        source_mask = []
        reference_mask = []
        reference_gt = []
        for idxx in range(batch_size):
            imgA = Image.open(os.path.join(args.source_dir, imga_name)).convert('RGB')
            imgA = transforms.Resize(256)(imgA)

            
            imgB_org = Image.open(os.path.join(args.reference_dir, imga_name.split('/')[-1])).convert('RGB')
            imgB = imgB_org
            imgB = transforms.Resize(torch.from_numpy(np.array(imgA)).shape[-2])(imgB)

            imgA_in, imgB_in, maskA_in, maskB_in = inference.transfer_test_data(imgA, imgB, postprocess=False)

            maskA_in = remake_masks(maskA_in)
            
            
            reference_grey.append(imgA_in[0])
            reference_jitter.append(imgB_in[0])
            source_mask.append(maskA_in[0])
            reference_mask.append(maskB_in[0])
            reference_gt.append(transform_gt(imgB_org))

        reference_grey = torch.stack(reference_grey).to(args.device)
        reference_jitter = torch.stack(reference_jitter).to(args.device)
        source_mask = torch.stack(source_mask).to(args.device)
        reference_mask = torch.stack(reference_mask).to(args.device)
        reference_gt = torch.stack(reference_gt).to(args.device)
        result = inference.transfer_test_calc(reference_grey, reference_jitter, source_mask, reference_mask)
        
        
        reference_mask = remake_masks(reference_mask)
        reference_mask_ = [reference_mask[:,0:1]*10., reference_mask[:,1:2]*40.,reference_mask[:,2:3]*100.,reference_mask[:,3:4]*160.,reference_mask[:,4:5]*200.,reference_mask[:,5:6]*1.]
        reference_mask_ = torch.cat(reference_mask_, dim=1)
        reference_mask_ = torch.sum(reference_mask_, dim=1, keepdim=True)
        reference_mask_ = torch.cat([reference_mask_, reference_mask_, reference_mask_], dim=1) / 255.
        
        source_mask_ = [source_mask[:,0:1]*10., source_mask[:,1:2]*40.,source_mask[:,2:3]*100.,source_mask[:,3:4]*160.,source_mask[:,4:5]*200.,source_mask[:,5:6]*1.]
        source_mask_ = torch.cat(source_mask_, dim=1)
        source_mask_ = torch.sum(source_mask_, dim=1, keepdim=True)
        source_mask_ = torch.cat([source_mask_, source_mask_, source_mask_], dim=1) / 255.
        
        if result is None:
            continue
        
        vis_train([reference_grey.cpu(),
                   reference_jitter.cpu(),
                   result.detach().cpu(),
                   reference_gt.cpu(),
                   source_mask_.cpu(),
                   reference_mask_.cpu()], i)





def vis_train(img_train_batch, idx):
    # saving training results
    img_train_batch = torch.cat(img_train_batch, dim=3)
    save_path = os.path.join(args.save_folder, '{:d}_fake.png'.format(idx))
    vis_image = make_grid(de_norm(img_train_batch), 1)
    save_image(vis_image, save_path)  # , normalize=True)

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def load_mask(path, img_size):
    mask = np.array(Image.open(path).convert('L'))
    mask = torch.FloatTensor(mask).unsqueeze(0)
    mask = functional.resize(mask, img_size, transforms.InterpolationMode.NEAREST)
    return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='test_ref')
    parser.add_argument("--train_phase", type=str, default='phase_1')
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model", 
                        default='')

    parser.add_argument("--source-dir", type=str, default="")
    parser.add_argument("--reference-dir", type=str, default="")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    config = get_config()
    main(config, args)