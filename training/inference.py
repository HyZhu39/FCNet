from typing import List
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from training.solver_phase1 import Solver as Solver_1
from training.solver_phase2 import Solver as Solver_2
from training.solver_phase3 import Solver as Solver_3
from training.preprocess import PreProcess
from models.modules.pseudo_gt import expand_area, mask_blend

class InputSample:
    def __init__(self, inputs, apply_mask=None):
        self.inputs = inputs
        self.transfer_input = None
        self.attn_out_list = None
        self.apply_mask = apply_mask

    def clear(self):
        self.transfer_input = None
        self.attn_out_list = None


class Inference:
    """
    An inference wrapper for makeup transfer.
    It takes two image `source` and `reference` in,
    and transfers the makeup of reference to source.
    """
    def __init__(self, config, args, model_path="G.pth"):

        self.device = args.device
        if args.train_phase == 'phase_1':
            Solver = Solver_1
        elif args.train_phase == 'phase_2':
            Solver = Solver_2
        elif args.train_phase == 'phase_3':
            Solver = Solver_3
        else:
            assert False
        self.solver = Solver(config, args, inference=model_path)
        self.preprocess = PreProcess(config, args.device)
        self.denoise = config.POSTPROCESS.WILL_DENOISE
        self.img_size = config.DATA.IMG_SIZE
        # TODO: can be a hyper-parameter
        self.eyeblur = {'margin': 12, 'blur_size':7}

    def prepare_input(self, *data_inputs):
        """
        data_inputs: List[image, mask, diff, lms]
        """
        inputs = []
        for i in range(len(data_inputs)):
            inputs.append(data_inputs[i].to(self.device).unsqueeze(0))
        # prepare mask
        inputs[1] = torch.cat((inputs[1][:,0:1], inputs[1][:,1:].sum(dim=1, keepdim=True)), dim=1)
        return inputs

    def postprocess(self, source, crop_face, result):
        if crop_face is not None:
            source = source.crop(
                (crop_face.left(), crop_face.top(), crop_face.right(), crop_face.bottom()))
        source = np.array(source)
        result = np.array(result)

        height, width = source.shape[:2]
        small_source = cv2.resize(source, (self.img_size, self.img_size))
        laplacian_diff = source.astype(
            np.float) - cv2.resize(small_source, (width, height)).astype(np.float)
        result = (cv2.resize(result, (width, height)) +
                  laplacian_diff).round().clip(0, 255)

        result = result.astype(np.uint8)

        if self.denoise:
            result = cv2.fastNlMeansDenoisingColored(result)
        result = Image.fromarray(result).convert('RGB')
        return result

    
    def generate_source_sample(self, source_input):
        """
        source_input: List[image, mask, diff, lms]
        """
        source_input = self.prepare_input(*source_input)
        return InputSample(source_input)

    def generate_reference_sample(self, reference_input, apply_mask=None, 
                                  source_mask=None, mask_area=None, saturation=1.0):
        """
        all the operations on the mask, e.g., partial mask, saturation, 
        should be finally defined in apply_mask
        """
        if source_mask is not None and mask_area is not None:
            apply_mask = self.generate_partial_mask(source_mask, mask_area, saturation)
            apply_mask = apply_mask.unsqueeze(0).to(self.device)
        reference_input = self.prepare_input(*reference_input)
        
        if apply_mask is None:
            apply_mask = torch.ones(1, 1, self.img_size, self.img_size).to(self.device)
        return InputSample(reference_input, apply_mask)


    def generate_partial_mask(self, source_mask, mask_area='full', saturation=1.0):
        """
        source_mask: (C, H, W), lip, face, left eye, right eye
        return: apply_mask: (1, H, W)
        """
        assert mask_area in ['full', 'skin', 'lip', 'eye']
        if mask_area == 'full':
            return torch.sum(source_mask[0:2], dim=0, keepdim=True) * saturation
        elif mask_area == 'lip':
            return source_mask[0:1] * saturation
        elif mask_area == 'skin':
            mask_l_eye = expand_area(source_mask[2:3], self.eyeblur['margin'])
            mask_r_eye = expand_area(source_mask[3:4], self.eyeblur['margin'])
            mask_eye = mask_l_eye + mask_r_eye
            mask_eye = mask_blend(mask_eye, 1.0, blur_size=self.eyeblur['blur_size'])
            return source_mask[1:2] * (1 - mask_eye) * saturation
        elif mask_area == 'eye':
            mask_l_eye = expand_area(source_mask[2:3], self.eyeblur['margin'])
            mask_r_eye = expand_area(source_mask[3:4], self.eyeblur['margin'])
            mask_eye = mask_l_eye + mask_r_eye
            mask_eye = mask_blend(mask_eye, saturation, blur_size=self.eyeblur['blur_size'])
            return mask_eye
  

    @torch.no_grad()
    def interface_transfer(self, source_sample: InputSample, reference_samples: List[InputSample]):
        """
        Input: a source sample and multiple reference samples
        Return: PIL.Image, the fused result
        """
        # encode source
        if source_sample.transfer_input is None:
            source_sample.transfer_input = self.solver.G.get_transfer_input(*source_sample.inputs)
        
        # encode references
        for r_sample in reference_samples:
            if r_sample.transfer_input is None:
                r_sample.transfer_input = self.solver.G.get_transfer_input(*r_sample.inputs, True)

        # self attention
        if source_sample.attn_out_list is None:
            source_sample.attn_out_list = self.solver.G.get_transfer_output(
                    *source_sample.transfer_input, *source_sample.transfer_input
                )
        
        # full transfer for each reference
        for r_sample in reference_samples:
            if r_sample.attn_out_list is None:
                r_sample.attn_out_list = self.solver.G.get_transfer_output(
                    *source_sample.transfer_input, *r_sample.transfer_input
                )

        # fusion
        fused_attn_out_list = []
        for i in range(len(source_sample.attn_out_list)):
            init_attn_out = torch.zeros_like(source_sample.attn_out_list[i], device=self.device)
            fused_attn_out_list.append(init_attn_out)
        apply_mask_sum = torch.zeros((1, 1, self.img_size, self.img_size), device=self.device)
        
        for r_sample in reference_samples:
            if r_sample.apply_mask is not None:
                apply_mask_sum += r_sample.apply_mask
                for i in range(len(source_sample.attn_out_list)):
                    feature_size = r_sample.attn_out_list[i].shape[2]
                    apply_mask = F.interpolate(r_sample.apply_mask, feature_size, mode='nearest')
                    fused_attn_out_list[i] += apply_mask * r_sample.attn_out_list[i]

        # self as reference
        source_apply_mask = 1 - apply_mask_sum.clamp(0, 1)
        for i in range(len(source_sample.attn_out_list)):
            feature_size = source_sample.attn_out_list[i].shape[2]
            apply_mask = F.interpolate(source_apply_mask, feature_size, mode='nearest')
            fused_attn_out_list[i] += apply_mask * source_sample.attn_out_list[i]

        # decode
        result = self.solver.G.decode(
            source_sample.transfer_input[0], fused_attn_out_list
        )
        result = self.solver.de_norm(result).squeeze(0)
        result = ToPILImage()(result.cpu())
        return result

    
    def transfer(self, source: Image, reference: Image, postprocess=True, source_mask_in=None):
        """
        Args:
            source (Image): The image where makeup will be transfered to.
            reference (Image): Image containing targeted makeup.
        Return:
            Image: Transfered image.
        """
        source_input, _, _ = self.preprocess(source)
        reference_input, _, _ = self.preprocess(reference)
        if not (source_input and reference_input):
            return None

        assert len(source_input)==len(reference_input)==4
        source_mask = source_input[1].unsqueeze(0)
        reference_mask = reference_input[1].unsqueeze(0)
        if source_mask_in is not None:
            source_mask = source_mask_in.unsqueeze(0)

        source_input = self.prepare_input(*source_input)
        reference_input = self.prepare_input(*reference_input)
        result = self.solver.test(source_input[0], reference_input[0], mask_s_full=source_mask, mask_r_full=reference_mask)
        
        if not postprocess:
            return result
        else:
            return self.postprocess(source, crop_face, result)
    
    def transfer_mul(self, source: Image, reference: Image, postprocess=True, source_mask_in=None):
        """
        Args:
            source (Image): The image where makeup will be transfered to.
            reference (Image): Image containing targeted makeup.
        Return:
            Image: Transfered image.
        """
        source_input, source_masks = self.preprocess.preprocess_test(source)
        reference_input, reference_masks = self.preprocess.preprocess_test(reference)

        source_mask =     source_masks.unsqueeze(0)        #source_input[1].unsqueeze(0)
        reference_mask =  reference_masks.unsqueeze(0)     #reference_input[1].unsqueeze(0)

        if source_mask_in is not None:
            source_mask = source_mask_in.unsqueeze(0)

        source_input =     source_input.to(self.device).unsqueeze(0)
        reference_input =  reference_input.to(self.device).unsqueeze(0)
        result, fake1, fake2, fake3, fake4, fake5 = self.solver.test(source_input, reference_input, mask_s_full=source_mask, mask_r_full=reference_mask)

        if not postprocess:
            return result, fake1, fake2, fake3, fake4, fake5
        else:
            return self.postprocess(source, crop_face, result)

    def transfer_mul_moretimes(self, source: Image, reference: Image):
        source_input, source_masks = self.preprocess.preprocess_test(source)
        reference_input, reference_masks = self.preprocess.preprocess_test(reference)
        source_mask =     source_masks.unsqueeze(0)
        reference_mask =  reference_masks.unsqueeze(0)
        source_input =     source_input.to(self.device).unsqueeze(0)
        reference_input =  reference_input.to(self.device).unsqueeze(0)
        
        z1 = torch.randn(source_input.shape[0], 512).to(self.device)
        z2 = torch.randn(source_input.shape[0], 512).to(self.device)
        z3 = torch.randn(source_input.shape[0], 512).to(self.device)
        z4 = torch.randn(source_input.shape[0], 512).to(self.device)
        z5 = torch.randn(source_input.shape[0], 512).to(self.device)
        
        results = []
        fake1s = []
        fake2s = []
        fake3s = []
        fake4s = []
        fake5s = []
        
        result, fake1, fake2, fake3, fake4, fake5 = self.solver.test_fix_z(source_input, reference_input, mask_s_full=source_mask, mask_r_full=reference_mask, \
                                                                     z1=z1 ,\
                                                                     z2=z2 ,\
                                                                     z3=z3 ,\
                                                                     z4=z4 ,\
                                                                     z5=z5 )
        results.append(result); fake1s.append(fake1); fake2s.append(fake2); fake3s.append(fake3); fake4s.append(fake4); fake5s.append(fake5)
        
        result, fake1, fake2, fake3, fake4, fake5 = self.solver.test_fix_z(source_input, reference_input, mask_s_full=source_mask, mask_r_full=reference_mask, \
                                                                     z1=z1 ,\
                                                                     z2=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z3=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z4=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z5=torch.randn(source_input.shape[0], 512).to(self.device) )
        results.append(result); fake1s.append(fake1); fake2s.append(fake2); fake3s.append(fake3); fake4s.append(fake4); fake5s.append(fake5)
        
        result, fake1, fake2, fake3, fake4, fake5 = self.solver.test_fix_z(source_input, reference_input, mask_s_full=source_mask, mask_r_full=reference_mask, \
                                                                     z1=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z2=z2 ,\
                                                                     z3=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z4=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z5=torch.randn(source_input.shape[0], 512).to(self.device) )
        results.append(result); fake1s.append(fake1); fake2s.append(fake2); fake3s.append(fake3); fake4s.append(fake4); fake5s.append(fake5)
        
        result, fake1, fake2, fake3, fake4, fake5 = self.solver.test_fix_z(source_input, reference_input, mask_s_full=source_mask, mask_r_full=reference_mask, \
                                                                     z1=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z2=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z3=z3 ,\
                                                                     z4=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z5=torch.randn(source_input.shape[0], 512).to(self.device) )
        results.append(result); fake1s.append(fake1); fake2s.append(fake2); fake3s.append(fake3); fake4s.append(fake4); fake5s.append(fake5)
        
        result, fake1, fake2, fake3, fake4, fake5 = self.solver.test_fix_z(source_input, reference_input, mask_s_full=source_mask, mask_r_full=reference_mask, \
                                                                     z1=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z2=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z3=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z4=z4 ,\
                                                                     z5=torch.randn(source_input.shape[0], 512).to(self.device) )
        results.append(result); fake1s.append(fake1); fake2s.append(fake2); fake3s.append(fake3); fake4s.append(fake4); fake5s.append(fake5)
        
        result, fake1, fake2, fake3, fake4, fake5 = self.solver.test_fix_z(source_input, reference_input, mask_s_full=source_mask, mask_r_full=reference_mask, \
                                                                     z1=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z2=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z3=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z4=torch.randn(source_input.shape[0], 512).to(self.device) ,\
                                                                     z5=z5 )
        results.append(result); fake1s.append(fake1); fake2s.append(fake2); fake3s.append(fake3); fake4s.append(fake4); fake5s.append(fake5)
        
        return results, fake1s, fake2s, fake3s, fake4s, fake5s
    
    def transfer_for_single_clustering(self, reference: Image):
        
        reference_input, reference_masks = self.preprocess.preprocess_test(reference)
        reference_mask =  reference_masks.unsqueeze(0)     #reference_input[1].unsqueeze(0)
        reference_input =  reference_input.to(self.device).unsqueeze(0) #self.prepare_input(*reference_input)
        
        z1 = torch.randn(reference_input.shape[0], 512).to(self.device)
        
        fake1 = self.solver.test_fix_z_single(reference_input, reference_mask, z1=z1)
        
        return z1, fake1
    
    def transfer_for_clustering_with_singleZ(self, reference: Image, z):
        
        reference_input, reference_masks = self.preprocess.preprocess_test(reference)
        reference_mask =  reference_masks.unsqueeze(0)
        reference_input =  reference_input.to(self.device).unsqueeze(0)
        
        z1 = z.to(self.device)
        
        fake1 = self.solver.test_fix_z_single(reference_input, reference_mask, z1=z1)
        
        return fake1
    
    def get_z_with_img(self, x, mask):
        z = self.solver.get_z_for_test(x, mask)
        return z
        
    
    def transfer_phase3(self, reference: Image):
        reference_input, reference_masks = self.preprocess.preprocess_test(reference)
        reference_mask =  reference_masks.unsqueeze(0)
        reference_input =  reference_input.to(self.device).unsqueeze(0)
        fake1 = self.solver.test(reference_input, reference_mask)
        return fake1
    
        
    def transfer_for_clustering_with_Zs(self, reference: Image, z1, z2, z3, z4, z5):
        
        reference_input, reference_masks = self.preprocess.preprocess_test(reference)
        reference_mask =  reference_masks.unsqueeze(0)
        reference_input =  reference_input.to(self.device).unsqueeze(0)
        
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)
        z3 = z3.to(self.device)
        z4 = z4.to(self.device)
        z5 = z5.to(self.device)
        
        fake1 = self.solver.test_fix_zs(reference_input, reference_mask, z1=z1, z2=z2, z3=z3, z4=z4, z5=z5)
        
        return fake1
    
    
    def transfer_with_another_img(self, imgA, imgB):
        
        imgA_grey, imgA_masks = self.preprocess.preprocess_test(imgA)
        imgA_mask =  imgA_masks.to(self.device).unsqueeze(0)
        imgA_grey =  imgA_grey.to(self.device).unsqueeze(0)
        imgB = imgB.to(self.device).unsqueeze(0)
        
        fake1 = self.solver.test_w_other_img(imgA_grey, imgA_mask, imgB)
        
        return fake1
    
    
    def get_mask(self, imgA):
        imgA_grey, imgA_masks = self.preprocess.preprocess_test(imgA)
        return imgA_grey, imgA_masks
    
    def transfer_with_another_mul_img(self, imgA, imgB1, imgB2, imgB3, imgB4, imgB5):
        
        imgA_grey, imgA_masks = self.preprocess.preprocess_test(imgA)
        imgA_mask =  imgA_masks.to(self.device).unsqueeze(0)
        imgA_grey =  imgA_grey.to(self.device).unsqueeze(0)
        
        imgB1 = imgB1.to(self.device).unsqueeze(0)
        imgB2 = imgB2.to(self.device).unsqueeze(0)
        imgB3 = imgB3.to(self.device).unsqueeze(0)
        imgB4 = imgB4.to(self.device).unsqueeze(0)
        imgB5 = imgB5.to(self.device).unsqueeze(0)
        
        fake1 = self.solver.test_w_other_mul_img(imgA_grey, imgA_mask, imgB1, imgB2, imgB3, imgB4, imgB5)
        
        return fake1
    
    
    def transfer_for_clustering_with_Zs_with_mask(self, reference: Image, z1, z2, z3, z4):
        
        reference_input, reference_masks = self.preprocess.preprocess_test(reference)
        reference_mask =  reference_masks.unsqueeze(0)
        reference_input =  reference_input.to(self.device).unsqueeze(0)
        
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)
        z3 = z3.to(self.device)
        z4 = z4.to(self.device)
        
        fake1, masks_x_re, masks_x_re_out = self.solver.test_fix_zs_with_mask(reference_input, reference_mask, z1=z1, z2=z2, z3=z3, z4=z4)
        
        return fake1, masks_x_re, masks_x_re_out
    
    def Get_Z_from_img(self, Image):
        Image = Image.to(self.device)
        z = self.solver.get_z_for_test(Image)
        return z
    
    
    def transfer_test_data(self, source: Image, reference: Image, postprocess=True, source_mask_in=None):
        source_input, source_mask = self.preprocess.preprocess_test(source)
        reference_input, reference_mask = self.preprocess.preprocess_test(reference)

        source_input = source_input.unsqueeze(0).to(self.device)
        reference_input = reference_input.unsqueeze(0).to(self.device)
        source_mask = source_mask.unsqueeze(0).to(self.device)
        reference_mask = reference_mask.unsqueeze(0).to(self.device)

        return source_input, reference_input, source_mask, reference_mask
    
    def transfer_test_data_mul(self, source, reference, postprocess=True, source_mask_in=None):
        source_input, source_mask = self.preprocess.preprocess_test(source)
        #reference_input, reference_mask = self.preprocess.preprocess_test(reference)
        source_input = source_input.unsqueeze(0).to(self.device)
        source_mask = source_mask.unsqueeze(0).to(self.device)
        
        reference_inputs = []
        reference_masks = []
        for i in reference:
            reference_input, reference_mask = self.preprocess.preprocess_test(i)
            reference_input = reference_input.unsqueeze(0).to(self.device)
            reference_mask = reference_mask.unsqueeze(0).to(self.device)
            reference_inputs.append(reference_input)
            reference_masks.append(reference_mask)

        return source_input, source_mask, reference_inputs, reference_masks
        
    def transfer_test_calc(self, source, reference, source_mask_in, reference_mask):
        return self.solver.test(source, reference, source_mask_in, reference_mask)
    
    def transfer_test_calc_mul(self, source, reference, source_mask_in, ref_mask_in):
        return self.solver.test_mul(source, reference, source_mask_in, ref_mask_in)
    
    def transfer_test_calc_vis(self, source, reference, source_mask_in):
        return self.solver.test_vis(source, reference, mask_s_full=source_mask_in, mask_r_full=source_mask_in)
    
    def joint_transfer(self, source: Image, reference_lip: Image, reference_skin: Image,
                       reference_eye: Image, postprocess=True):
        source_input, face, crop_face = self.preprocess(source)
        lip_input, _, _ = self.preprocess(reference_lip)
        skin_input, _, _ = self.preprocess(reference_skin)
        eye_input, _, _ = self.preprocess(reference_eye)
        if not (source_input and lip_input and skin_input and eye_input):
            return None

        source_mask = source_input[1]
        source_sample = self.generate_source_sample(source_input)
        reference_samples = [
            self.generate_reference_sample(lip_input, source_mask=source_mask, mask_area='lip'),
            self.generate_reference_sample(skin_input, source_mask=source_mask, mask_area='skin'),
            self.generate_reference_sample(eye_input, source_mask=source_mask, mask_area='eye')
        ]
        
        result = self.interface_transfer(source_sample, reference_samples)
        
        if not postprocess:
            return result
        else:
            return self.postprocess(source, crop_face, result)