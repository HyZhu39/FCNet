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

from models.model import get_discriminator, vgg16, get_generator_alter
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

def merge_sty_codes(sty_codes, STYLECODE_DIM):
    '''
    sty_codes:  [N, STYLECODE_DIM * (num_domains+1), 1, 1]
    '''
    _,C,_,_ = sty_codes[0].shape
    assert C == 5 * STYLECODE_DIM
    output = sty_codes[0].clone()
    output[:,STYLECODE_DIM * 1:STYLECODE_DIM * 2] = sty_codes[1][:,STYLECODE_DIM * 1:STYLECODE_DIM * 2]
    output[:,STYLECODE_DIM * 2:STYLECODE_DIM * 3] = sty_codes[2][:,STYLECODE_DIM * 2:STYLECODE_DIM * 3]
    output[:,STYLECODE_DIM * 3:STYLECODE_DIM * 4] = sty_codes[3][:,STYLECODE_DIM * 3:STYLECODE_DIM * 4]
    output[:,STYLECODE_DIM * 4:STYLECODE_DIM * 5] = sty_codes[4][:,STYLECODE_DIM * 4:STYLECODE_DIM * 5]
    return output

def gradient_loss(gen_frames, gt_frames, alpha=1):

    def gradient(x):
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = right - left, bottom - top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)
    
class Solver(): 
    def __init__(self, config, args, logger=None, inference=False):
        self.G = get_generator_alter(config)
        self.domain_num = config.MODEL.DOMAINS
        self.style_dim = config.MODEL.STYLECODE_DIM
        
        self.epoch = 1

        if inference:
            self.G.load_state_dict(torch.load(inference, map_location='cpu'))
            self.G = self.G.to(args.device).eval()
            return
        self.double_d = config.TRAINING.DOUBLE_D
        self.D_A = get_discriminator(config)
        if self.double_d:
            self.D_B = get_discriminator(config)

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
        if self.pgt_annealing:
            self.pgt_maker = AnnealingComposePGT(self.margins,
                                                 config.PGT.SKIN_ALPHA_MILESTONES, config.PGT.SKIN_ALPHA_VALUES,
                                                 config.PGT.EYE_ALPHA_MILESTONES, config.PGT.EYE_ALPHA_VALUES,
                                                 config.PGT.LIP_ALPHA_MILESTONES, config.PGT.LIP_ALPHA_VALUES,
                                                 config.PGT.HAIR_ALPHA_MILESTONES, config.PGT.HAIR_ALPHA_VALUES
                                                 )
        else:
            self.pgt_maker = ComposePGT(self.margins,
                                        config.PGT.SKIN_ALPHA,
                                        config.PGT.EYE_ALPHA,
                                        config.PGT.LIP_ALPHA,
                                        config.PGT.HAIR_ALPHA
                                        )
        self.pgt_maker.eval()

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

        self.lambda_hair = config.LOSS.LAMBDA_MAKEUP_HAIR
        self.lambda_background = config.LOSS.LAMBDA_MAKEUP_BACKGROUND
        # style
        self.lambda_style = config.LOSS.LAMBDA_STYLE
        self.lambda_grad = config.LOSS.LAMBDA_GRAD
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
            'G-loss-grad': [],
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
        self.D_A.apply(self.weights_init_xavier)
        if self.double_d:
            self.D_B.apply(self.weights_init_xavier)

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(gan_mode='lsgan')
        self.criterionPGT = MakeupLoss()
        self.vgg = vgg16(pretrained=True)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_A_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), self.d_lr,
                                              [self.beta1, self.beta2])
        if self.double_d:
            self.d_B_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), self.d_lr,
                                                  [self.beta1, self.beta2])
        
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer,
                                                                      T_max=self.num_epochs,
                                                                      eta_min=self.g_lr * self.lr_decay_factor)
        self.d_A_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_A_optimizer,
                                                                        T_max=self.num_epochs,
                                                                        eta_min=self.d_lr * self.lr_decay_factor)
        if self.double_d:
            self.d_B_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_B_optimizer,
                                                                            T_max=self.num_epochs,
                                                                            eta_min=self.d_lr * self.lr_decay_factor)


        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D_A, 'D_A')
        if self.double_d: self.print_network(self.D_B, 'D_B')

        self.G.to(self.device)
        self.vgg.to(self.device)
        self.D_A.to(self.device)
        if self.double_d: self.D_B.to(self.device)
        
        if self.keepon:
            self.load_checkpoint()

    def train(self, data_loader):
        self.len_dataset = len(data_loader)

        for epoch_now in range(self.epoch, self.num_epochs + 1):
            self.epoch = epoch_now
            self.start_time = time.time()
            loss_tmp = self.get_loss_tmp()
            self.G.train()

            self.D_A.train()
            if self.double_d: self.D_B.train()
            losses_G = []
            losses_D_A = []
            losses_D_B = []
            losses_sty = []
            losses_ds = []
            losses_grad = []
            losses_pgt = []

            with tqdm(data_loader, desc="training") as pbar:
                for step, (
                        reference_gt, reference_grey, mask, reference_jitters, reference_jitter_greys, reference_jitter_masks, reference_jitters_asgt, reference_jitter_gt, reference_jitter_gt_grey, reference_jitter_gt_mask, reference_jitter_gt_asgt, reference_jitter_gt_asgt_for_fakeB
                        ) in enumerate(pbar):
                    reference_gt = reference_gt.to(self.device)
                    reference_grey = reference_grey.to(self.device)
                    reference_mask = mask.to(self.device)
                    reference_jitters = reference_jitters.to(self.device)
                    reference_jitter_greys = reference_jitter_greys.to(self.device)
                    reference_jitter_masks = reference_jitter_masks.to(self.device)
                    reference_jitter_gt = reference_jitter_gt.to(self.device)
                    reference_jitter_gt_grey = reference_jitter_gt_grey.to(self.device)
                    reference_jitter_gt_mask = reference_jitter_gt_mask.to(self.device)

                    reference_jitters_asgt = reference_jitters_asgt.to(self.device)
                    reference_jitter_gt_asgt = reference_jitter_gt_asgt.to(self.device)
                    
                    reference_jitter_gt_asgt_for_fakeB = reference_jitter_gt_asgt_for_fakeB.to(self.device)
                    
                    reference_jitters = reference_jitters.permute(1,0,2,3,4)
                    reference_jitter_greys = reference_jitter_greys.permute(1,0,2,3,4)
                    reference_jitter_masks = reference_jitter_masks.permute(1,0,2,3,4)
                    reference_jitters_asgt = reference_jitters_asgt.permute(1,0,2,3,4)

                    # ================= Generate ================== #
                    fake_outs_imgs = []
                    fake_outs_stys = []
                    for i in range(self.domain_num + 1):
                        fake_img, fake_sty = self.G(reference_jitter_greys[i], reference_jitters[i], reference_mask, reference_jitter_masks[i], True)
                        fake_outs_imgs.append(fake_img)
                        fake_outs_stys.append(fake_sty)
                        
                    fin_sty = merge_sty_codes(fake_outs_stys, self.style_dim)

                    fake_A = self.G.forward_with_colorcode(reference_jitter_gt_grey, fin_sty, reference_mask)

                    fake_B = self.G(reference_jitter_gt, reference_jitter_gt_grey, reference_jitter_gt_mask, reference_mask)
                    
                    fake_A_gt = self.G(reference_grey, reference_gt, reference_mask, reference_mask)

                    # ================== Train D ================== #
                    # Real
                    out = self.D_A(reference_jitter_gt)
                    d_loss_real1 = self.criterionGAN(out, True)
                    out = self.D_A(reference_gt)
                    d_loss_real2 = self.criterionGAN(out, True)
                    d_loss_real = (d_loss_real1 + d_loss_real2)
                    # Fake
                    out = self.D_A(fake_A.detach().contiguous())
                    d_loss_fake1 = self.criterionGAN(out, False)
                    out = self.D_A(fake_A_gt.detach().contiguous())
                    d_loss_fake2 = self.criterionGAN(out, False)
                    d_loss_fake = (d_loss_fake1 + d_loss_fake2)

                    # Backward + Optimize
                    d_loss = (d_loss_real + d_loss_fake) * 0.5
                    self.d_A_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_A_optimizer.step()

                    # Logging
                    loss_tmp['D-A-loss_real'] += d_loss_real.item()
                    loss_tmp['D-A-loss_fake'] += d_loss_fake.item()
                    losses_D_A.append(d_loss.item())

                    # training D_B, D_B aims to distinguish class A
                    # Real
                    if self.double_d:
                        out = self.D_B(reference_jitter_gt_grey)
                    else:
                        out = self.D_A(reference_jitter_gt_grey)
                    d_loss_real = self.criterionGAN(out, True)
                    # Fake
                    if self.double_d:
                        out = self.D_B(fake_B.detach().contiguous())
                    else:
                        out = self.D_A(fake_B.detach().contiguous())
                    d_loss_fake = self.criterionGAN(out, False)

                    # Backward + Optimize
                    d_loss = (d_loss_real + d_loss_fake) * 0.5
                    if self.double_d:
                        self.d_B_optimizer.zero_grad()
                        d_loss.backward()
                        self.d_B_optimizer.step()
                    else:
                        self.d_A_optimizer.zero_grad()
                        d_loss.backward()
                        self.d_A_optimizer.step()

                    # Logging
                    loss_tmp['D-B-loss_real'] += d_loss_real.item()
                    loss_tmp['D-B-loss_fake'] += d_loss_fake.item()
                    losses_D_B.append(d_loss.item())

                    # ================== Train G ================== #

                    idt_A = self.G(reference_jitter_gt_grey, reference_jitter_gt_grey, reference_mask, reference_mask)
                    idt_B = self.G(reference_jitter_gt, reference_jitter_gt, reference_jitter_gt_mask, reference_jitter_gt_mask)

                    loss_idt_A = self.criterionL1(idt_A, reference_jitter_gt_grey) * self.lambda_A * self.lambda_idt
                    loss_idt_B = self.criterionL1(idt_B, reference_jitter_gt) * self.lambda_B * self.lambda_idt

                    # loss_idt
                    loss_idt = (loss_idt_A + loss_idt_B) * 0.5

                    # loss_style loss

                    loss_style = torch.tensor([0],dtype=torch.float32, device=self.device)
                    for j in range(self.domain_num + 1):
                        loss_style += self.criterionPGT(fake_outs_imgs[j], reference_jitters_asgt[j], reference_mask[:, 0:1]) * self.lambda_lip
                        loss_style += self.criterionPGT(fake_outs_imgs[j], reference_jitters_asgt[j], reference_mask[:, 1:2]) * self.lambda_skin
                        loss_style += self.criterionPGT(fake_outs_imgs[j], reference_jitters_asgt[j], reference_mask[:, 2:3]) * self.lambda_eye
                        loss_style += self.criterionPGT(fake_outs_imgs[j], reference_jitters_asgt[j], reference_mask[:, 3:4]) * self.lambda_hair
                        loss_style += self.criterionPGT(fake_outs_imgs[j], reference_jitters_asgt[j], reference_mask[:, 4:5]) * self.lambda_background
                    loss_style = loss_style * self.lambda_style * self.lambda_A
                    
                    # loss_grad loss
                    if self.epoch <= 5:
                        loss_grad = torch.tensor([0]).to(self.device)
                    else:
                        loss_grad = gradient_loss(reference_jitter_gt_grey, fake_A) * self.lambda_grad * self.lambda_A

                    # loss ds
                    
                    loss_ds = self.criterionL1(fake_A_gt, reference_gt) * self.lambda_ds * self.lambda_A

                    # GAN loss D_A(G_A(A))
                    pred_fake = self.D_A(fake_A)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True)
                    pred_fake = self.D_A(fake_A_gt)
                    g_A_loss_adv += self.criterionGAN(pred_fake, True)

                    # GAN loss D_B(G_B(B))
                    if self.double_d:
                        pred_fake = self.D_B(fake_B)
                    else:
                        pred_fake = self.D_A(fake_B)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)

                    g_A_loss_pgt = 0
                    g_B_loss_pgt = 0

                    mask_lip = reference_mask[:, 0:1]
                    mask_lip_B = reference_jitter_gt_mask[:, 0:1]
                    g_A_lip_loss_pgt = self.criterionPGT(fake_A, reference_jitter_gt_asgt, mask_lip) * self.lambda_lip
                    g_B_lip_loss_pgt = self.criterionPGT(fake_B, reference_jitter_gt_asgt_for_fakeB, mask_lip_B) * self.lambda_lip
                    g_A_loss_pgt = g_A_lip_loss_pgt + g_A_loss_pgt
                    g_B_loss_pgt = g_B_lip_loss_pgt + g_B_loss_pgt

                    mask_eye = reference_mask[:, 2:3].sum(dim=1, keepdim=True)
                    mask_eye_B = reference_jitter_gt_mask[:, 2:3].sum(dim=1, keepdim=True)
                    g_A_eye_loss_pgt = self.criterionPGT(fake_A, reference_jitter_gt_asgt, mask_eye) * self.lambda_eye
                    g_B_eye_loss_pgt = self.criterionPGT(fake_B, reference_jitter_gt_asgt_for_fakeB, mask_eye_B) * self.lambda_eye
                    g_A_loss_pgt = g_A_eye_loss_pgt + g_A_loss_pgt
                    g_B_loss_pgt = g_B_eye_loss_pgt + g_B_loss_pgt

                    mask_skin = reference_mask[:, 1:2]
                    mask_skin_B = reference_jitter_gt_mask[:, 1:2]
                    g_A_skin_loss_pgt = self.criterionPGT(fake_A, reference_jitter_gt_asgt, mask_skin) * self.lambda_skin
                    g_B_skin_loss_pgt = self.criterionPGT(fake_B, reference_jitter_gt_asgt_for_fakeB, mask_skin_B) * self.lambda_skin
                    g_A_loss_pgt = g_A_skin_loss_pgt + g_A_loss_pgt
                    g_B_loss_pgt = g_B_skin_loss_pgt + g_B_loss_pgt

                    mask_hair = reference_mask[:, 3:4]
                    mask_hair_B = reference_jitter_gt_mask[:, 3:4]
                    g_A_hair_loss_pgt = self.criterionPGT(fake_A, reference_jitter_gt_asgt, mask_hair) * self.lambda_hair
                    g_B_hair_loss_pgt = self.criterionPGT(fake_B, reference_jitter_gt_asgt_for_fakeB, mask_hair_B) * self.lambda_hair
                    g_A_loss_pgt = g_A_hair_loss_pgt + g_A_loss_pgt
                    g_B_loss_pgt = g_B_hair_loss_pgt + g_B_loss_pgt
                    
                    # BACKGROUND
                    mask_background = reference_mask[:, 4:5]
                    mask_background_B = reference_jitter_gt_mask[:, 4:5]
                    if self.lambda_background != 0:
                        g_A_background_loss_pgt = self.criterionPGT(fake_A, reference_jitter_gt_asgt,
                                                                    mask_background) * self.lambda_background
                        g_B_background_loss_pgt = self.criterionPGT(fake_B, reference_jitter_gt_asgt_for_fakeB,
                                                                    mask_background_B) * self.lambda_background
                        g_A_loss_pgt += g_A_background_loss_pgt
                        g_B_loss_pgt += g_B_background_loss_pgt
                    else:
                        g_A_background_loss_pgt = torch.tensor([0]).to(self.device)
                        g_B_background_loss_pgt = torch.tensor([0]).to(self.device)
                        

                    rec_A = self.G(fake_A, reference_grey, reference_mask, reference_mask)
                    rec_B = self.G(fake_B, reference_jitter_gt, reference_jitter_gt_mask, reference_jitter_gt_mask)

                    g_loss_rec_A = (self.criterionL1(rec_A, reference_grey)) * self.lambda_A
                    g_loss_rec_B = (self.criterionL1(rec_B, reference_jitter_gt)) * self.lambda_B

                    # vgg loss
                    vgg_s = self.vgg(reference_jitter_gt_asgt).detach()
                    vgg_fake_A = self.vgg(fake_A)
                    g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_s) * self.lambda_A * self.lambda_vgg

                    vgg_r = self.vgg(reference_jitter_gt_asgt_for_fakeB).detach()
                    vgg_fake_B = self.vgg(fake_B)
                    g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_r) * self.lambda_B * self.lambda_vgg

                    loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5

                    # Combined loss
                    g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_pgt + g_B_loss_pgt + \
                             + loss_style + loss_ds + loss_grad

                    self.g_optimizer.zero_grad()  # may the multi opt be the crash reason?
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss_tmp['G-A-loss-adv'] += g_A_loss_adv.item()
                    loss_tmp['G-B-loss-adv'] += g_B_loss_adv.item()
                    loss_tmp['G-loss-idt'] += loss_idt.item()
                    loss_tmp['G-loss-img-rec'] += (g_loss_rec_A + g_loss_rec_B).item() * 0.5
                    loss_tmp['G-loss-vgg-rec'] += (g_loss_A_vgg + g_loss_B_vgg).item() * 0.5
                    loss_tmp['G-loss-rec'] += loss_rec.item()
                    loss_tmp['G-loss-skin-pgt'] += (g_A_skin_loss_pgt + g_B_skin_loss_pgt).item()
                    loss_tmp['G-loss-eye-pgt'] += (g_A_eye_loss_pgt + g_B_eye_loss_pgt).item()
                    loss_tmp['G-loss-lip-pgt'] += (g_A_lip_loss_pgt + g_B_lip_loss_pgt).item()
                    loss_tmp['G-loss-hair-pgt'] += (g_A_hair_loss_pgt + g_B_hair_loss_pgt).item()
                    loss_tmp['G-loss-background-pgt'] += (g_A_background_loss_pgt + g_B_background_loss_pgt).item()
                    loss_tmp['G-loss-style'] += loss_style.item()
                    loss_tmp['G-loss-self'] += loss_ds.item()
                    loss_tmp['G-loss-grad'] += loss_grad.item()

                    loss_tmp['G-loss-pgt'] += (g_A_loss_pgt + g_B_loss_pgt).item()
                    losses_G.append(g_loss.item())
                    losses_sty.append(loss_style.item())
                    losses_ds.append(loss_ds.item())
                    losses_grad.append(loss_grad.item())
                    losses_pgt.append((g_A_loss_pgt + g_B_loss_pgt).item())
                    pbar.set_description(
                        "Epoch: %d, Step: %d, Loss_G: %0.4f, Loss_A: %0.4f, Loss_B: %0.4f, Loss_grad: %0.4f, Loss_style: %0.4f , Loss_self: %0.4f, Loss_pgt: %0.4f" % \
                        (self.epoch, step + 1, np.mean(losses_G), np.mean(losses_D_A), np.mean(losses_D_B),
                         np.mean(losses_grad), np.mean(losses_sty), np.mean(losses_ds), np.mean(losses_pgt)))

                    # save the images
                    if step % self.vis_step_freq == 0:
                        self.vis_train_step([reference_grey.detach().cpu(),
                                             reference_jitters[0].detach().cpu(),
                                             reference_jitters[1].detach().cpu(),
                                             reference_jitters[2].detach().cpu(),
                                             reference_jitters[3].detach().cpu(),
                                             reference_jitters[4].detach().cpu(),
                                             fake_outs_imgs[0].detach().cpu(),
                                             fake_outs_imgs[1].detach().cpu(),
                                             fake_outs_imgs[2].detach().cpu(),
                                             fake_outs_imgs[3].detach().cpu(),
                                             fake_outs_imgs[4].detach().cpu(),
                                             reference_jitter_gt.detach().cpu(),
                                             reference_jitter_gt_asgt.detach().cpu(),
                                             fake_A.detach().cpu(),
                                             reference_gt.detach().cpu()], step)

            self.end_time = time.time()
            for k, v in loss_tmp.items():
                loss_tmp[k] = v / self.len_dataset
            loss_tmp['G-loss'] = np.mean(losses_G)
            loss_tmp['D-A-loss'] = np.mean(losses_D_A)
            loss_tmp['D-B-loss'] = np.mean(losses_D_B)
            self.log_loss(loss_tmp)
            self.plot_loss()

            # Decay learning rate
            self.g_scheduler.step()
            self.d_A_scheduler.step()
            if self.double_d:
                self.d_B_scheduler.step()

            if self.pgt_annealing:
                self.pgt_maker.step()

            # save the images
            if (self.epoch) % self.vis_freq == 0:
                fake_outs_imgs_test = []
                fake_outs_stys_test = []
                with torch.no_grad():
                    for i in range(self.domain_num + 1):
                        fake_img, fake_sty = self.G(reference_jitter_greys[i][0:1], reference_jitters[i][0:1], reference_mask[0:1], reference_jitter_masks[i][0:1], True)
                        fake_outs_imgs_test.append(fake_img)
                        fake_outs_stys_test.append(fake_sty)
                    fin_sty_test = merge_sty_codes(fake_outs_stys_test, self.style_dim)
                    test = self.G.forward_with_colorcode(reference_jitter_gt_grey[0:1], fin_sty_test, reference_mask[0:1])
                self.vis_train([reference_jitter_gt_grey[0:1].detach().cpu(),
                                reference_jitter_gt[0:1].detach().cpu(),
                                test.detach().cpu(),
                                reference_gt[0:1].detach().cpu()])  # ,

            # Save model checkpoints
            if (self.epoch) % self.save_freq == 0:
                self.save_models()

    def get_loss_tmp(self):
        loss_tmp = {
            'D-A-loss_real': 0.0,
            'D-A-loss_fake': 0.0,
            'D-B-loss_real': 0.0,
            'D-B-loss_fake': 0.0,
            'G-A-loss-adv': 0.0,
            'G-B-loss-adv': 0.0,
            'G-loss-idt': 0.0,
            'G-loss-img-rec': 0.0,
            'G-loss-vgg-rec': 0.0,
            'G-loss-rec': 0.0,
            'G-loss-skin-pgt': 0.0,
            'G-loss-eye-pgt': 0.0,
            'G-loss-lip-pgt': 0.0,
            'G-loss-hair-pgt': 0.0,
            'G-loss-background-pgt': 0.0,
            'G-loss-pgt': 0.0,
            'G-loss-style': 0.0,
            'G-loss-grad': 0.0,
            'G-loss-self': 0.0,
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
        D_A_losses = []
        D_A_names = []
        D_B_losses = []
        D_B_names = []
        D_P_losses = []
        D_P_names = []
        for k, v in self.loss_logger.items():
            if 'G' in k:
                G_names.append(k)
                G_losses.append(v)
            elif 'D-A' in k:
                D_A_names.append(k)
                D_A_losses.append(v)
            elif 'D-B' in k:
                D_B_names.append(k)
                D_B_losses.append(v)
            elif 'D-P' in k:
                D_P_names.append(k)
                D_P_losses.append(v)
        plot_curves(self.save_folder, 'G_loss', G_losses, G_names, ylabel='Loss')
        plot_curves(self.save_folder, 'D-A_loss', D_A_losses, D_A_names, ylabel='Loss')
        plot_curves(self.save_folder, 'D-B_loss', D_B_losses, D_B_names, ylabel='Loss')

    def load_checkpoint(self):
        G_path = os.path.join(self.load_folder, 'G.pth')
        if os.path.exists(G_path):
            self.G.load_state_dict(torch.load(G_path, map_location=self.device))
            self.g_optimizer.load_state_dict(torch.load(os.path.join(self.load_folder, 'opt.pth'), map_location=self.device))
            self.epoch = int(self.load_folder.split('/')[-1].replace('epoch_','')) + 1
            print('loaded trained generator {}..!'.format(G_path))
        D_A_path = os.path.join(self.load_folder, 'D_A.pth')
        if os.path.exists(D_A_path):
            self.D_A.load_state_dict(torch.load(D_A_path, map_location=self.device))
            self.d_A_optimizer.load_state_dict(torch.load(os.path.join(self.load_folder, 'D_opt.pth'), map_location=self.device))
            print('loaded trained discriminator A {}..!'.format(D_A_path))

        if self.double_d:
            D_B_path = os.path.join(self.load_folder, 'D_B.pth')
            if os.path.exists(D_B_path):
                self.D_B.load_state_dict(torch.load(D_B_path, map_location=self.device))
                self.d_B_optimizer.load_state_dict(torch.load(os.path.join(self.load_folder, 'DB_opt.pth'), map_location=self.device))
                print('loaded trained discriminator B {}..!'.format(D_B_path))

    def save_models(self):
        save_dir = os.path.join(self.save_folder, 'epoch_{:d}'.format(self.epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.G.state_dict(), os.path.join(save_dir, 'G.pth'))
        torch.save(self.D_A.state_dict(), os.path.join(save_dir, 'D_A.pth'))
        torch.save(self.g_optimizer.state_dict(), os.path.join(save_dir, 'opt.pth'))
        torch.save(self.d_A_optimizer.state_dict(), os.path.join(save_dir, 'D_opt.pth'))
        if self.double_d:
            torch.save(self.D_B.state_dict(), os.path.join(save_dir, 'D_B.pth'))
            torch.save(self.d_B_optimizer.state_dict(), os.path.join(save_dir, 'DB_opt.pth'))

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

    def generate(self, image_A, image_B, mask_s, mask_r):
        """image_A is content, image_B is style"""
        mask_s = mask_s.to(image_B.device)
        mask_r = mask_r.to(image_B.device)

        with torch.no_grad():
            fake_img, fake_sty = self.G(image_A, image_B, mask_s, mask_r, True)
            test = self.G.forward_with_colorcode(image_A, fake_sty, mask_s)
        
        return test

    def test(self, image_A, image_B, mask_s, mask_r):
        self.G.eval()
        fake_A = self.generate(image_A, image_B, mask_s, mask_r)
        fake_A_org = fake_A
        fake_A = self.de_norm(fake_A)

        return fake_A_org

    def test_mul(self, image_A, image_Bs, source_mask, ref_masks):
        self.G.eval()
        fake_A = self.generate_mul(image_A, image_Bs, source_mask, ref_masks)
        fake_A_org = fake_A
        return fake_A_org
        
    def generate_mul(self, image_A, image_Bs, source_mask, ref_masks):
        """image_A is content, image_B is style"""
        
        fake_outs_imgs_test = []
        fake_outs_stys_test = []

        for i in range(self.domain_num + 1):
            fake_img, fake_sty = self.G(image_A, image_Bs[i], source_mask.to(image_A.device), ref_masks[i].to(image_Bs[i].device), True)
            fake_outs_imgs_test.append(fake_img)
            fake_outs_stys_test.append(fake_sty)            
        fin_sty = merge_sty_codes(fake_outs_stys_test, self.style_dim)
            
        with torch.no_grad():
            test = self.G.forward_with_colorcode(image_A, fin_sty, source_mask)
        return test
        

    def test_vis(self, image_A, image_B, mask_s_full, mask_r_full):
        self.G.eval()
        fake_A, ref_feat, feat_y, after_RSIM, feature_after_spades0, feature_after_spades1 = self.G.forward_vis(image_A, image_B, mask_s_full,
                                   mask_r_full)
        fake_A_org = fake_A
        fake_A = self.de_norm(fake_A)
        return fake_A_org, ref_feat, feat_y, after_RSIM, feature_after_spades0, feature_after_spades1
    
    
