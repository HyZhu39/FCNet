import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image, make_grid
import torch.nn.init as init
from tqdm import tqdm

from models.modules.pseudo_gt import expand_area
from models.model import get_discriminator, vgg16, get_auto_Z_Generator_wo_flow
from models.loss import GANLoss, MakeupLoss, ComposePGT, AnnealingComposePGT

from training.utils import plot_curves

from training.vgg_loss import PerceptualLoss, LPIPSloss


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

class Solver():
    def __init__(self, config, args, logger=None, inference=False):

        self.G = get_auto_Z_Generator_wo_flow(config)

        self.domain_num = config.MODEL.DOMAINS
        self.Gen_path = config.TRAINING.DIRECT_PHASE3_loadpath

        self.epoch = 1

        if inference:
            self.G.load_state_dict(torch.load(inference, map_location=args.device))
            self.G = self.G.to(args.device).eval()

            return

        self.load_folder = args.load_folder
        self.save_folder = args.save_folder
        self.vis_folder = os.path.join(args.save_folder, 'visualization')
        if not os.path.exists(self.vis_folder):
            os.makedirs(self.vis_folder)
        self.vis_freq = config.LOG.VIS_FREQ
        self.save_freq = config.LOG.SAVE_FREQ

        self.vis_step_freq = config.LOG.VIS_FREQ_STEP

        # Data & PGT
        self.img_size = config.DATA.IMG_SIZE

        # Hyper-param
        self.num_epochs = config.TRAINING.NUM_EPOCHS
        self.g_lr = config.TRAINING.G_LR
        self.d_lr = config.TRAINING.D_LR
        self.beta1 = config.TRAINING.BETA1
        self.beta2 = config.TRAINING.BETA2
        self.lr_decay_factor = config.TRAINING.LR_DECAY_FACTOR
        
        self.loss_l2_weight = config.LOSS.LAMBDA_L2
        self.loss_perceptual_weight = config.LOSS.LAMBDA_PERCEP
        self.loss_LPIPS_weight = config.LOSS.LAMBDA_LPIPS
        self.loss_color_weight = config.LOSS.LAMBDA_COLOR

        self.device = args.device
        self.keepon = args.keepon
        self.logger = logger
        self.loss_logger = {
            'G-loss-L1': [],
            'G-loss-L2': [],
            'G-loss-color': [],
            'G-loss-perceptual': [],
            'G-loss-LPIPS': [],
            'G-loss': [],
        }

        self.build_model()
        super(Solver, self).__init__()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        if self.logger is not None:
            self.logger.info('{:s}, the number of parameters: {:d}'.format(name, num_params))
        else:
            print('{:s}, the number of parameters: {:d}'.format(name, num_params))

    # For generator
    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data, gain=1.0)

    def build_model(self):
        self.G.apply(self.weights_init_xavier)
        self.G.Generator.load_state_dict(torch.load(self.Gen_path, map_location=self.device))
        
        if self.keepon:
            self.load_checkpoint()

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        ###
        self.criterion_perceptual = PerceptualLoss(layer_weights={'conv5_4': 1.}, 
                                                   vgg_type='vgg19',
                                                   use_input_norm=True,
                                                   perceptual_weight=1.0,
                                                   style_weight=0.,
                                                   norm_img=True,  # -1,1 -> 0,1
                                                   criterion='l1'
                                                   ).to(self.device)
        self.criterionLPIPS = LPIPSloss(loss_weight=1.0, net_type='alex').to(self.device)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.auto_Z_Encoder.parameters(), lr=self.g_lr) # 1e-3

        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer,
                                                                      T_max=self.num_epochs,
                                                                      eta_min=self.g_lr * self.lr_decay_factor)

        # Print networks
        self.print_network(self.G, 'G')

        self.G.to(self.device)

    def train(self, data_loader):
        self.len_dataset = len(data_loader)

        for epoch_now in range(self.epoch, self.num_epochs + 1):
            self.epoch = epoch_now
            self.start_time = time.time()
            loss_tmp = self.get_loss_tmp()
            self.G.train()
            #self.G.lock_net_param(self.G.FLOW_generator)
            #self.G.FLOW_generator.eval()
            self.G.lock_net_param(self.G.Generator)
            self.G.Generator.eval()
            
            losses_G = []
            losses_L1 = []
            losses_L2 = []
            losses_color = []
            losses_perceptual = []
            losses_LPIPS = []
            
            with tqdm(data_loader, desc="training") as pbar:
                for step, (reference_gt, reference_grey, mask, grey__, mask__, gt__) in enumerate(pbar):

                    mask_gt = mask.to(self.device)  # (b, c', h, w)

                    reference_gt = reference_gt.to(self.device)
                    reference_grey = reference_grey.to(self.device)
                    grey = grey__.to(self.device)
                    mask = mask__.to(self.device)
                    gt = gt__.to(self.device)

                    # ================= Generate ================== #

                    fake_A = self.G(x=grey, mask=mask)

                    # ================== Train G ================== #

                    l1_loss = self.criterionL1(fake_A, gt)
                    l2_loss = torch.tensor([0],dtype=torch.float32, device=self.device)
                    percep_loss, _ = self.criterion_perceptual(fake_A, gt)
                    LPIPS_loss = self.criterionLPIPS(self.de_norm(fake_A), self.de_norm(gt))
                    
                    color_loss = self.color_loss(self.de_norm(fake_A) * 255.)
                    color_loss = torch.mean(color_loss)
                    
                    g_loss = l1_loss + self.loss_l2_weight * l2_loss + self.loss_color_weight * color_loss + self.loss_perceptual_weight * percep_loss + self.loss_LPIPS_weight * LPIPS_loss

                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss_tmp['G-loss'] += g_loss.item()
                    losses_G.append(g_loss.item())
                    
                    loss_tmp['G-loss-L1'] += l1_loss.item()
                    losses_L1.append(l1_loss.item())
                    
                    loss_tmp['G-loss-L2'] += l2_loss.item()
                    losses_L2.append(l2_loss.item())
                    
                    loss_tmp['G-loss-color'] += color_loss.item()
                    losses_color.append(color_loss.item())
                    
                    loss_tmp['G-loss-perceptual'] += percep_loss.item()
                    losses_perceptual.append(percep_loss.item())
                    
                    loss_tmp['G-loss-LPIPS'] += LPIPS_loss.item()
                    losses_LPIPS.append(LPIPS_loss.item())

                    pbar.set_description(
                        "Epoch: %d, Step: %d, Loss_G: %0.4f, Loss_L1: %0.4f, Loss_Percep: %0.4f, Loss_LPIPS: %0.4f, Loss_Color: %0.4f" % \
                        (self.epoch, step + 1, np.mean(losses_G), np.mean(losses_L1), np.mean(losses_perceptual), np.mean(losses_LPIPS), np.mean(losses_color)))

                    # save the images
                    if step % self.vis_step_freq == 0:
                        self.vis_train_step([grey.detach().cpu(),
                                             fake_A.detach().cpu(),
                                             gt.detach().cpu(),
                                             reference_grey.detach().cpu(),
                                             reference_gt.detach().cpu()], step)

            self.end_time = time.time()
            for k, v in loss_tmp.items():
                loss_tmp[k] = v / self.len_dataset
            loss_tmp['G-loss'] = np.mean(losses_G)
            loss_tmp['G-loss-L1'] = np.mean(losses_L1)
            loss_tmp['G-loss-L2'] = np.mean(losses_L2)
            loss_tmp['G-loss-color'] = np.mean(losses_color)
            loss_tmp['G-loss-perceptual'] = np.mean(losses_perceptual)
            loss_tmp['G-loss-LPIPS'] = np.mean(losses_LPIPS)
            
            self.log_loss(loss_tmp)

            # Decay learning rate
            self.g_scheduler.step()

            # # save the images
            if (self.epoch) % self.vis_freq == 0:
                with torch.no_grad():
                    fake = self.G(x=reference_grey[0:1], mask=mask_gt[0:1])
                self.vis_train([reference_grey[0:1].detach().cpu(),
                                fake.detach().cpu(),
                                reference_gt[0:1].detach().cpu()])  # ,


            # Save model checkpoints
            if (self.epoch) % self.save_freq == 0:
                self.save_models()

    def get_loss_tmp(self):
        loss_tmp = {
            'G-loss': 0.0,
            'G-loss-L1': 0.0,
            'G-loss-L2': 0.0,
            'G-loss-color': 0.0,
            'G-loss-perceptual': 0.0,
            'G-loss-LPIPS': 0.0,
        }
        return loss_tmp

    def log_loss(self, loss_tmp):
        if self.logger is not None:
            self.logger.info('\n' + '=' * 40 + '\nEpoch {:d}, time {:.2f} s'
                             .format(self.epoch, self.end_time - self.start_time))
        else:
            print('\n' + '=' * 40 + '\nEpoch {:d}, time {:d} s'
                  .format(self.epoch, self.end_time - self.start_time))
        for k, v in loss_tmp.items():
            self.loss_logger[k].append(v)
            if self.logger is not None:
                self.logger.info('{:s}\t{:.6f}'.format(k, v))
            else:
                print('{:s}\t{:.6f}'.format(k, v))
        if self.logger is not None:
            self.logger.info('=' * 40)
        else:
            print('=' * 40)

    def plot_loss(self):
        G_losses = []
        G_names = []
        for k, v in self.loss_logger.items():
            if 'G' in k:
                G_names.append(k)
                G_losses.append(v)
        plot_curves(self.save_folder, 'G_loss', G_losses, G_names, ylabel='Loss')

    def load_checkpoint(self):
        G_path = os.path.join(self.load_folder, 'G.pth')
        if os.path.exists(G_path):
            self.G.load_state_dict(torch.load(G_path, map_location=self.device))
            self.g_optimizer.load_state_dict(torch.load(os.path.join(self.load_folder, 'opt.pth'), map_location=self.device))
            self.epoch = int(self.load_folder.replace('epoch_',''))
            print('loaded trained generator {}..!'.format(G_path))

    def save_models(self):
        save_dir = os.path.join(self.save_folder, 'epoch_{:d}'.format(self.epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.G.state_dict(), os.path.join(save_dir, 'G.pth'))
        torch.save(self.g_optimizer.state_dict(), os.path.join(save_dir, 'opt.pth'))


    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def vis_train(self, img_train_batch):
        # saving training results
        img_train_batch = torch.cat(img_train_batch, dim=3)
        save_path = os.path.join(self.vis_folder, 'epoch_{:d}_fake.png'.format(self.epoch))
        vis_image = make_grid(self.de_norm(img_train_batch), 1)
        save_image(vis_image, save_path)  # , normalize=True)

    def vis_train_step(self, img_train_batch, step):
        # saving training results
        img_train_batch = torch.cat(img_train_batch, dim=3)
        base_sav_path = os.path.join(self.vis_folder, str(self.epoch))
        if not os.path.exists(base_sav_path):
            os.makedirs(base_sav_path)
        save_path = os.path.join(self.vis_folder, str(self.epoch), 'step_{:d}_fake.png'.format(step))
        vis_image = make_grid(self.de_norm(img_train_batch), 1)
        save_image(vis_image, save_path)  # , normalize=True)

    def generate(self, image_A, image_B, mask_A=None, mask_B=None,
                 diff_A=None, diff_B=None, lms_A=None, lms_B=None, mask_s_full=None, mask_r_full=None):
        """image_A is content, image_B is style"""
        with torch.no_grad():
            mask_r_full = mask_r_full.to(image_B.device)
            mask_s_full = mask_s_full.to(image_B.device)
            style_code_ref = self.StyleEncoder(image_B, mask_r_full)  # torch.Size([N, 4, 64])
            YH_YL = self.StyleDecoder(style_code_ref, mask_s_full)
            # res = self.G(image_A, image_B, mask_A, mask_B, diff_A, diff_B, lms_A, lms_B, YH_YL)
            # res = self.G(image_A, mask_A, diff_A, lms_A, YH_YL)
            res = self.G(image_A, YH_YL, mask_A)
        return res

    def test_org(self, image_A, mask_A, diff_A, lms_A, image_B, mask_B, diff_B, lms_B, mask_s_full, mask_r_full):
        with torch.no_grad():
            fake_A = self.generate(image_A, image_B, mask_A, mask_B, diff_A, diff_B, lms_A, lms_B, mask_s_full,
                                   mask_r_full)
        fake_A = self.de_norm(fake_A)
        fake_A = fake_A.squeeze(0)
        return ToPILImage()(fake_A.cpu())

    def test(self, x, mask):
        mask = remake_masks(mask).to(x.device)
        with torch.no_grad():
            fake_A = self.G(x, mask)
        fake_A = self.de_norm(fake_A)
        return fake_A
        #fake_A = fake_A.squeeze(0)
        #return ToPILImage()(fake_A.cpu())
        
    def color_loss(self, x):
        '''
        Input: torch Tensor NCHW, 0-255
        Output: color scores N 1 
        '''
        # split the image into its respective RGB components
        R, G, B = x.unbind(1)  # N 3 H W --> N H W

        # compute rg = R - G
        rg = torch.absolute(R - G)
 
        # compute yb = 0.5 * (R + G) - B
        # could be optimized by doing a >> 1 ?
        yb = torch.absolute(0.5 * (R + G) - B)
    
        #print('rg', rg.shape) # N H W
        #print('yb', yb.shape) # N H W
    
        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (torch.mean(rg, dim=[1,2]), torch.std(rg, dim=[1,2]))
        (ybMean, ybStd) = (torch.mean(yb, dim=[1,2]), torch.std(yb, dim=[1,2]))
     
        # combine the mean and standard deviations
        stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))
 
        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)
 


