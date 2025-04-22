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
np.random.seed(39)
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

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Inference(config, args, args.load_path)

    n_imgname = sorted(os.listdir(args.source_dir))

    m_imgname = sorted(os.listdir(args.reference_dir))
    
    for i, (imga_name, imgb_name) in enumerate(zip(n_imgname, m_imgname)):
        reference_grey = []
        reference_jitter1 = []
        reference_jitter2 = []
        reference_jitter3 = []
        reference_jitter4 = []
        reference_jitter5 = []
        source_mask = [] 
        ref_mask1 = []
        ref_mask2 = []
        ref_mask3 = []
        ref_mask4 = []
        ref_mask5 = []
        
        reference_gt = []
        for idxx in range(batch_size):
            imgA = Image.open(os.path.join(args.source_dir, imga_name)).convert('RGB')
            imgA = transforms.Resize(256)(imgA)
            
            imgB_org = Image.open(os.path.join(args.reference_dir, imga_name.split('/')[-1])).convert('RGB') 

            imgBs = []
            idx_refs = 5
            for _ in range(idx_refs):
                idx = np.random.randint(0,len(m_imgname)) # [low, high)
                imgB = Image.open(os.path.join(args.reference_dir, sorted(os.listdir(args.reference_dir))[idx] )).convert('RGB')
                imgB = transforms.Resize(torch.from_numpy(np.array(imgA)).shape[-2])(imgB)
                imgB = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)(imgB)                                                      # if without jitter
                imgBs.append(imgB)

            imgA_in, maskA_in, imgB_ins, maskB_ins = inference.transfer_test_data_mul(imgA, imgBs, postprocess=False)
            
            for t in range(len(maskB_ins)):
                maskB_ins[t] = remake_masks(maskB_ins[t])

            maskA_in = remake_masks(maskA_in)
            
            reference_grey.append(imgA_in[0])
            reference_jitter1.append(imgB_ins[0])
            reference_jitter2.append(imgB_ins[1])
            reference_jitter3.append(imgB_ins[2])
            reference_jitter4.append(imgB_ins[3])
            reference_jitter5.append(imgB_ins[4])
            source_mask.append(maskA_in[0])
            ref_mask1.append(maskB_ins[0])
            ref_mask2.append(maskB_ins[1])
            ref_mask3.append(maskB_ins[2])
            ref_mask4.append(maskB_ins[3])
            ref_mask5.append(maskB_ins[4])
            reference_gt.append(transform_gt(imgB_org))

        reference_grey = torch.stack(reference_grey).to(args.device)
        reference_jitter1 = torch.stack(reference_jitter1).to(args.device)[0]
        reference_jitter2 = torch.stack(reference_jitter2).to(args.device)[0]
        reference_jitter3 = torch.stack(reference_jitter3).to(args.device)[0]
        reference_jitter4 = torch.stack(reference_jitter4).to(args.device)[0]
        reference_jitter5 = torch.stack(reference_jitter5).to(args.device)[0]
        reference_jitters = [reference_jitter1,reference_jitter2,reference_jitter3,reference_jitter4,reference_jitter5]
        
        source_mask = torch.stack(source_mask).to(args.device)
        ref_mask1 = torch.stack(ref_mask1).to(args.device)[0]
        ref_mask2 = torch.stack(ref_mask2).to(args.device)[0]
        ref_mask3 = torch.stack(ref_mask3).to(args.device)[0]
        ref_mask4 = torch.stack(ref_mask4).to(args.device)[0]
        ref_mask5 = torch.stack(ref_mask5).to(args.device)[0]

        ref_masks = [ref_mask1,ref_mask2,ref_mask3,ref_mask4,ref_mask5]
        
        reference_gt = torch.stack(reference_gt).to(args.device)
        
        result = inference.transfer_test_calc_mul(reference_grey, reference_jitters, source_mask, ref_masks)
        
        if result is None:
            continue
        
        vis_train([reference_grey.cpu(),
                   reference_jitter1.cpu(),
                   reference_jitter2.cpu(),
                   reference_jitter3.cpu(),
                   reference_jitter4.cpu(),
                   reference_jitter5.cpu(),
                   result.detach().cpu(),
                   reference_gt.cpu()], i)





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
    parser.add_argument("--name", type=str, default='test_mul_ref')
    parser.add_argument("--train_phase", type=str, default='phase_1')
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model", 
                        default='')

    parser.add_argument("--source-dir", type=str, default="") #
    parser.add_argument("--reference-dir", type=str, default="") #
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    config = get_config()
    main(config, args)