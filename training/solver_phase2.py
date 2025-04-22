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
from models.model import get_discriminator, vgg16, get_flow_generator_alter
from models.loss import GANLoss, MakeupLoss, ComposePGT, AnnealingComposePGT

from training.utils import plot_curves

def remake_masks(masks):
    '''
    masks:     N domain_num HW
    # lip skin left_eye right_eye hair
    '''
    masks[:, 2:3] = masks[:, 2:4].sum(dim=1, keepdim=True)
    masks[:, 2:3][masks[:, 2:3] > 1] = 1
    masks = torch.concat((masks[:, 0:3], masks[:, 4:]), dim=1)
    masks[masks > 1] = 1

    mask_background = masks.sum(dim=1, keepdim=True)
    mask_background[mask_background != 0] = -1
    mask_background += 1

    masks = torch.cat([masks, mask_background], dim=1)
    return masks

class Solver():
    def __init__(self, config, args, logger=None, inference=False):
        self.G = get_flow_generator_alter(config)

        self.domain_num = config.MODEL.DOMAINS
        self.Gen_path = config.TRAINING.PHASE2_loadpath

        self.epoch = 1

        if inference:
            self.G.load_state_dict(torch.load(inference + '/G.pth', map_location=args.device))
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
        self.margins = {'eye': config.PGT.EYE_MARGIN,
                        'lip': config.PGT.LIP_MARGIN}
        self.pgt_annealing = config.PGT.ANNEALING


        # Hyper-param
        self.num_epochs = config.TRAINING.NUM_EPOCHS
        self.g_lr = config.TRAINING.G_LR
        self.d_lr = config.TRAINING.D_LR
        self.beta1 = config.TRAINING.BETA1
        self.beta2 = config.TRAINING.BETA2
        self.lr_decay_factor = config.TRAINING.LR_DECAY_FACTOR

        # Loss param
        self.lambda_idt = config.LOSS.LAMBDA_IDT
        self.lambda_A = config.LOSS.LAMBDA_A
        self.lambda_B = config.LOSS.LAMBDA_B
        self.lambda_lip = config.LOSS.LAMBDA_MAKEUP_LIP
        self.lambda_skin = config.LOSS.LAMBDA_MAKEUP_SKIN
        self.lambda_eye = config.LOSS.LAMBDA_MAKEUP_EYE
        self.lambda_vgg = config.LOSS.LAMBDA_VGG

        # NEW
        self.lambda_hair = config.LOSS.LAMBDA_MAKEUP_HAIR
        self.lambda_background = config.LOSS.LAMBDA_MAKEUP_BACKGROUND
        # style
        self.lambda_style = config.LOSS.LAMBDA_STYLE
        self.lambda_ds = config.LOSS.LAMBDA_DS

        self.device = args.device
        self.keepon = args.keepon
        self.logger = logger
        self.loss_logger = {
            'D-A-loss_real': [],
            'D-A-loss_fake': [],
            'D-B-loss_real': [],
            'D-B-loss_fake': [],
            'G-A-loss-adv': [],
            'G-B-loss-adv': [],
            'G-loss-idt': [],
            'G-loss-img-rec': [],
            'G-loss-vgg-rec': [],
            'G-loss-rec': [],
            'G-loss-skin-pgt': [],
            'G-loss-eye-pgt': [],
            'G-loss-lip-pgt': [],
            'G-loss-hair-pgt': [],
            'G-loss-background-pgt': [],
            'G-loss-pgt': [],
            'G-loss-self': [],
            'G-loss-style': [],
            'G-loss': [],
            'D-A-loss': [],
            'D-B-loss': []
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
        self.criterionGAN = GANLoss(gan_mode='lsgan')
        self.criterionPGT = MakeupLoss()
        self.vgg = vgg16(pretrained=True)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.inn.parameters(), lr=self.g_lr) # 1e-3

        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer,
                                                                      T_max=self.num_epochs,
                                                                      eta_min=self.g_lr * self.lr_decay_factor)

        # Print networks
        self.print_network(self.G, 'G')

        self.G.to(self.device)
        self.vgg.to(self.device)

    def train(self, data_loader):
        self.len_dataset = len(data_loader)

        for epoch_now in range(self.epoch, self.num_epochs + 1):
            self.epoch = epoch_now
            self.start_time = time.time()
            loss_tmp = self.get_loss_tmp()
            self.G.train()
            self.G.lock_net_param(self.G.Generator)
            self.G.Generator.eval()

            losses_G = []

            with tqdm(data_loader, desc="training") as pbar:
                for step, (
                        reference_gt, reference_grey, mask, jitters_, mask_, jitter_wo_aug) in enumerate(pbar):

                    mask = mask.to(self.device)
                    mask_y = mask_.to(self.device)

                    reference_gt = reference_gt.to(self.device)
                    reference_grey = reference_grey.to(self.device)
                    reference_jitter = jitters_.to(self.device)
                    jitter_wo_aug = jitter_wo_aug.to(self.device)

                    # ================= Generate ================== #
                    z, log_jac_det, fake_A = self.G(x=reference_grey, masks_x=mask, y=reference_jitter, y_mask=mask_y, is_Training=True)
                    # ================== Train G ================== #

                    loss = 0.5 * torch.sum(z ** 2, 1).to(self.device) - log_jac_det
                    g_loss = loss.mean() / self.G.flow_dim

                    self.g_optimizer.zero_grad()  # may the multi opt be the crash reason?
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss_tmp['G-loss'] += g_loss.item()
                    losses_G.append(g_loss.item())

                    pbar.set_description(
                        "Epoch: %d, Step: %d, Loss_G: %0.4f" % \
                        (self.epoch, step + 1, np.mean(losses_G)))

                    # save the images
                    if step % self.vis_step_freq == 0:
                        z = torch.randn(reference_grey.shape[0], self.G.flow_dim).to(self.device)
                        fake_colored = self.G(x=reference_grey, masks_x=mask, z=z)
                        self.vis_train_step([reference_grey.detach().cpu(),
                                             reference_jitter.detach().cpu(),
                                             jitter_wo_aug.detach().cpu(),
                                             fake_colored.detach().cpu(),
                                             fake_A.detach().cpu(),
                                             reference_gt.detach().cpu()], step)

            self.end_time = time.time()
            for k, v in loss_tmp.items():
                loss_tmp[k] = v / self.len_dataset
            loss_tmp['G-loss'] = np.mean(losses_G)
            self.log_loss(loss_tmp)

            # Decay learning rate
            self.g_scheduler.step()

            # Save model checkpoints
            if (self.epoch) % self.save_freq == 0:
                self.save_models()

    def get_loss_tmp(self):
        loss_tmp = {
            'G-loss': 0.0,
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
            style_code_ref = self.StyleEncoder(image_B, mask_r_full)
            YH_YL = self.StyleDecoder(style_code_ref, mask_s_full)
            res = self.G(image_A, YH_YL, mask_A)
        return res

    def test_org(self, image_A, mask_A, diff_A, lms_A, image_B, mask_B, diff_B, lms_B, mask_s_full, mask_r_full):
        with torch.no_grad():
            fake_A = self.generate(image_A, image_B, mask_A, mask_B, diff_A, diff_B, lms_A, lms_B, mask_s_full,
                                   mask_r_full)
        fake_A = self.de_norm(fake_A)
        fake_A = fake_A.squeeze(0)
        return ToPILImage()(fake_A.cpu())

    def generate_rand(self, image_A, image_B, mask_A=None, mask_B=None,
                      diff_A=None, diff_B=None, lms_A=None, lms_B=None, mask_s_full=None, mask_r_full=None):
        """image_A is content, image_B is style"""
        with torch.no_grad():
            mask_r_full = mask_r_full.to(image_B.device)
            mask_s_full = mask_s_full.to(image_B.device)
            style_code_ref = self.StyleEncoder(image_B, mask_r_full)  # torch.Size([N, 4, 64])

            style_code_ref[:, 1:2] = torch.randn([1, 1, 64])

            YH_YL = self.StyleDecoder(style_code_ref, mask_s_full)
            res = self.G(image_A, YH_YL, mask_A)
        return res

    def test(self, image_A, image_B, mask_s_full, mask_r_full):
        z1 = torch.randn(image_A.shape[0], self.G.flow_dim).to(image_A.device)
        z2 = torch.randn(image_A.shape[0], self.G.flow_dim).to(image_A.device)
        z3 = torch.randn(image_A.shape[0], self.G.flow_dim).to(image_A.device)
        z4 = torch.randn(image_A.shape[0], self.G.flow_dim).to(image_A.device)
        z5 = torch.randn(image_A.shape[0], self.G.flow_dim).to(image_A.device)
        mask_s_full = mask_s_full.to(image_A.device)
        with torch.no_grad():
            # fake_colored = self.G(x=reference_grey, masks_x=mask_r_full, z=z)
            fake_1 = self.G(x=image_A, masks_x=mask_s_full, z=z1)
            fake_2 = self.G(x=image_A, masks_x=mask_s_full, z=z2)
            fake_3 = self.G(x=image_A, masks_x=mask_s_full, z=z3)
            fake_4 = self.G(x=image_A, masks_x=mask_s_full, z=z4)
            fake_5 = self.G(x=image_A, masks_x=mask_s_full, z=z5)
            fake_A = self.G.test(x=image_A, masks_x=mask_s_full, z1=z1, z2=z2, z3=z3, z4=z4, z5=z5)
        fake_A = self.de_norm(fake_A)
        fake_A = fake_A.squeeze(0)
        return ToPILImage()(fake_A.cpu()), \
            ToPILImage()(self.de_norm(fake_1).cpu().squeeze(0)), \
            ToPILImage()(self.de_norm(fake_2).cpu().squeeze(0)), \
            ToPILImage()(self.de_norm(fake_3).cpu().squeeze(0)), \
            ToPILImage()(self.de_norm(fake_4).cpu().squeeze(0)), \
            ToPILImage()(self.de_norm(fake_5).cpu().squeeze(0))
    
    def test_fix_z(self, image_A, image_B, mask_s_full, mask_r_full, z1, z2, z3, z4, z5):
        z1 = z1.to(image_A.device)
        z2 = z2.to(image_A.device)
        z3 = z3.to(image_A.device)
        z4 = z4.to(image_A.device)
        z5 = z5.to(image_A.device)
        
        
        mask_s_full = mask_s_full.to(image_A.device)
        with torch.no_grad():
            # fake_colored = self.G(x=reference_grey, masks_x=mask_r_full, z=z)
            fake_1 = self.G(x=image_A, masks_x=mask_s_full, z=z1)
            fake_2 = self.G(x=image_A, masks_x=mask_s_full, z=z2)
            fake_3 = self.G(x=image_A, masks_x=mask_s_full, z=z3)
            fake_4 = self.G(x=image_A, masks_x=mask_s_full, z=z4)
            fake_5 = self.G(x=image_A, masks_x=mask_s_full, z=z5)
            fake_A = self.G.test(x=image_A, masks_x=mask_s_full, z1=z1, z2=z2, z3=z3, z4=z4, z5=z5)
        fake_A = self.de_norm(fake_A)
        fake_A = fake_A.squeeze(0)
        return ToPILImage()(fake_A.cpu()), \
            ToPILImage()(self.de_norm(fake_1).cpu().squeeze(0)), \
            ToPILImage()(self.de_norm(fake_2).cpu().squeeze(0)), \
            ToPILImage()(self.de_norm(fake_3).cpu().squeeze(0)), \
            ToPILImage()(self.de_norm(fake_4).cpu().squeeze(0)), \
            ToPILImage()(self.de_norm(fake_5).cpu().squeeze(0))
    
    def test_fix_z_single(self, image_B, mask_s_full, z1):

        z1 = z1.to(image_B.device)
        
        mask_s_full = mask_s_full.to(image_B.device)
        mask_s_full = remake_masks(mask_s_full)
        
        with torch.no_grad():
            fake_1 = self.G.test_w_single_z(x=image_B, masks_x=mask_s_full, z1=z1)

        return fake_1 

    def test_fix_latent_single(self, image_B, mask_s_full, latent):

        mask_s_full = mask_s_full.to(image_B.device)
        mask_s_full = remake_masks(mask_s_full)
        
        with torch.no_grad():
            fake_1 = self.G.test_w_single_latent(x=image_B, masks_x=mask_s_full, z1=z1)

        return fake_1 

    def test_fix_zs(self, image_B, mask_s_full, z1, z2, z3, z4, z5):

        z1 = z1.to(image_B.device)
        z2 = z2.to(image_B.device)
        z3 = z3.to(image_B.device)
        z4 = z4.to(image_B.device)
        z5 = z5.to(image_B.device)
        
        mask_s_full = mask_s_full.to(image_B.device)
        mask_s_full = remake_masks(mask_s_full)
        
        with torch.no_grad():
            fake_1,out1,out2,out3,out4,out5 = self.G.test(x=image_B, masks_x=mask_s_full, z1=z1, z2=z2, z3=z3, z4=z4, z5=z5)

        return fake_1 ,out1,out2,out3,out4,out5

    

    def test_fix_zs_with_mask(self, image_B, mask_s_full, z1, z2, z3, z4):

        z1 = z1.to(image_B.device)
        z2 = z2.to(image_B.device)
        z3 = z3.to(image_B.device)
        z4 = z4.to(image_B.device)
        
        mask_s_full = mask_s_full.to(image_B.device)
        with torch.no_grad():
            fake_1, masks_x_re, masks_x_re_out = self.G.test_with_mask(x=image_B, masks_x=mask_s_full, z1=z1, z2=z2, z3=z3, z4=z4)

        fake_1 = self.de_norm(fake_1)
        fake_1 = fake_1.squeeze(0) #CHW
        return ToPILImage()(fake_1.cpu()), masks_x_re, masks_x_re_out
    
    def get_z_for_test(self, image_B, mask):        
        mask = remake_masks(mask)
        image_B = image_B # .to(image_B.device)
        mask = mask #.to(image_B.device)
        
        with torch.no_grad():
            fake_y_sty = self.G.Generator.Get_y_sty(image_B, mask) #.squeeze(-1).squeeze(-1)  # N dim
            fake_y_sty = fake_y_sty.squeeze(-1).squeeze(-1)
            z, _ = self.G.inn(fake_y_sty)
        return z
    
    
    def test_w_other_img(self, imgA_grey, imgA_mask, imgB):        

        fake_y_sty = self.G.Generator.Get_y_sty(imgB) #.squeeze(-1).squeeze(-1)  # N dim
        fake_y_sty = fake_y_sty.squeeze(-1).squeeze(-1)
        fake_colored = self.G.forward_w_diff_img(imgA_grey, imgA_mask, fake_y_sty)    
        return fake_colored
    
    def test_w_other_mul_img(self, imgA_grey, imgA_mask, imgB1, imgB2, imgB3, imgB4, imgB5):        

        fake_y_sty1 = self.G.Generator.Get_y_sty(imgB1) #.squeeze(-1).squeeze(-1)  # N dim
        fake_y_sty1 = fake_y_sty1.squeeze(-1).squeeze(-1)
        fake_y_sty2 = self.G.Generator.Get_y_sty(imgB2)
        fake_y_sty2 = fake_y_sty2.squeeze(-1).squeeze(-1)
        fake_y_sty3 = self.G.Generator.Get_y_sty(imgB3)
        fake_y_sty3 = fake_y_sty3.squeeze(-1).squeeze(-1)
        fake_y_sty4 = self.G.Generator.Get_y_sty(imgB4)
        fake_y_sty4 = fake_y_sty4.squeeze(-1).squeeze(-1)
        fake_y_sty5 = self.G.Generator.Get_y_sty(imgB5)
        fake_y_sty5 = fake_y_sty5.squeeze(-1).squeeze(-1)
        fake_colored = self.G.forward_w_diff_mul_img(imgA_grey, imgA_mask, fake_y_sty1, fake_y_sty2, fake_y_sty3, fake_y_sty4, fake_y_sty5)    
        return fake_colored