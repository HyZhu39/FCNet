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
Z_num = 1 #500

# only one can be True
TEST_W_random_Z = True #False
TEST_W_single_Z = False
TEST_W_mul_Z = False #True
Z_path = 'z/'

transform = transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
            
def remake_masks(masks):
    '''
    masks:     N domain_num HW
    # lip skin left_eye right_eye hair
    '''
    masks[:, 2:3] = masks[:, 2:4].sum(dim=1, keepdim=True)
    masks[:, 2:3][masks[:, 2:3] > 1] = 1                              # attention this!!!
    masks = torch.concat((masks[:, 0:3], masks[:, 4:]), dim=1) 
    masks[masks > 1] = 1                                              # ensure no overlap

    mask_background = masks.sum(dim=1, keepdim=True)  # torch.zeros_like(masks[:, 0:1, :, :]) # N 1 256 256
    mask_background[mask_background != 0] = -1
    mask_background += 1

    masks = torch.cat([masks, mask_background], dim=1)
    return masks


def vis_train(img_train_batch, idx, sav_path):
    # saving training results
    img_train_batch = torch.cat(img_train_batch, dim=3)
    save_path = os.path.join(sav_path, '{:d}_fake.png'.format(idx)) # args.save_folder
    vis_image = make_grid(de_norm(img_train_batch), 1)
    save_image(vis_image, save_path)  # , normalize=True)

def vis_train_single(img, idx, sav_path):
    # saving training results
    save_path = os.path.join(sav_path, '{:d}_fake.png'.format(idx)) # args.save_folder
    vis_image = de_norm(img)
    save_image(vis_image, save_path)

def vis_train_single_ref(img, idx, sav_path):
    # saving training results
    save_path = os.path.join(sav_path, '{:d}_ref.png'.format(idx)) # args.save_folder
    vis_image = de_norm(img)
    save_image(vis_image, save_path)

def sav_tensor(tensor, idx, sav_path):
    save_path = os.path.join(sav_path, '{:d}.pth'.format(idx))
    torch.save(tensor, save_path)

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def load_mask(path, img_size):
    mask = np.array(Image.open(path).convert('L'))
    mask = torch.FloatTensor(mask).unsqueeze(0)
    mask = functional.resize(mask, img_size, transforms.InterpolationMode.NEAREST)
    return mask

def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Inference(config, args, args.load_path)

    n_imgname = sorted(os.listdir(args.source_dir))
    m_imgname = sorted(os.listdir(args.reference_dir))
    
    gt_path = ''
    
    
    for i in range(len(n_imgname)):
        
        imgA = Image.open(os.path.join(args.source_dir, n_imgname[i])).convert('RGB')
        imgA = transforms.Resize(RESIZE_SIZE)(imgA)
        imgA_vis = transform(imgA).unsqueeze(0)
        
        #reference_gt = Image.open(os.path.join(gt_path, n_imgname[i])).convert('RGB')
        #reference_gt = transforms.Resize(RESIZE_SIZE)(reference_gt)
        #reference_gt = transform(reference_gt).unsqueeze(0)
        
        print(imgA.size)
        
        if os.path.exists(os.path.join(args.save_folder, str(i))):
            sav_path = os.path.join(args.save_folder, str(i))
        else:
            os.mkdir(os.path.join(args.save_folder, str(i)))
            sav_path = os.path.join(args.save_folder, str(i))
            
        if os.path.exists(os.path.join(sav_path, 'single')):
            sav_path_single = os.path.join(os.path.join(sav_path, 'single'))
        else:
            os.mkdir(os.path.join(os.path.join(sav_path, 'single')))
            sav_path_single = os.path.join(os.path.join(sav_path, 'single'))
            
        if os.path.exists(os.path.join(sav_path, 'z')):
            sav_path_z = os.path.join(os.path.join(sav_path, 'z'))
        else:
            os.mkdir(os.path.join(os.path.join(sav_path, 'z')))
            sav_path_z = os.path.join(os.path.join(sav_path, 'z'))
        
        if TEST_W_random_Z:
            for idxx in range(Z_num):
                z1 = torch.randn(1, 320)  # 64*5=320
                result_1 = inference.transfer_for_clustering_with_singleZ(imgA, z1).to('cpu')

                vis_train([imgA_vis.cpu(),
                       result_1.detach().cpu(),
                       ], idxx, sav_path) # reference_gt.cpu()
            
                vis_train_single(result_1.detach().cpu(), idxx, sav_path_single)
                sav_tensor(z1, idxx, sav_path_z)
        elif TEST_W_single_Z:
            z_names = sorted(os.listdir(Z_path))
            for idxx in range(len(z_names)):
                z1 = torch.load(os.path.join(Z_path, str(idxx)+'.pth'))
                result_1 = inference.transfer_for_clustering_with_singleZ(imgA, z1).to('cpu')

                vis_train([imgA_vis.cpu(),
                       result_1.detach().cpu(),
                       ], idxx, sav_path) # reference_gt.cpu()
            
                vis_train_single(result_1.detach().cpu(), idxx, sav_path_single) 
        elif TEST_W_mul_Z:
            z_names = sorted(os.listdir(Z_path))
            for idxx in range(len(z_names)):
                z_lips = torch.load(os.path.join(Z_path, str((idxx+0)%len(z_names))+'.pth'))
                z_skin = torch.load(os.path.join(Z_path, str((idxx+1)%len(z_names))+'.pth'))
                z_eyes = torch.load(os.path.join(Z_path, str((idxx+2)%len(z_names))+'.pth'))
                z_hair = torch.load(os.path.join(Z_path, str((idxx+3)%len(z_names))+'.pth'))
                z_bkgd = torch.load(os.path.join(Z_path, str((idxx+4)%len(z_names))+'.pth'))
                
                result_all,result_1,result_2,result_3,result_4,result_5 = inference.transfer_for_clustering_with_Zs(imgA, z_lips, z_skin, z_eyes, z_hair, z_bkgd)

                vis_train([imgA_vis.cpu(),
                       result_1.detach().cpu(),
                       result_2.detach().cpu(),
                       result_3.detach().cpu(),
                       result_4.detach().cpu(),
                       result_5.detach().cpu(),
                       result_all.detach().cpu(),
                       ], idxx, sav_path) # reference_gt.cpu()
            
                vis_train_single(result_all.detach().cpu(), idxx, sav_path_single) 

            
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='phase_2')
    parser.add_argument("--train_phase", type=str, default='phase_2')
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
