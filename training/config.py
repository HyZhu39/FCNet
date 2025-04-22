from fvcore.common.config import CfgNode

"""
This file defines default options of configurations.
It will be further merged by yaml files and options from
the command-line.
Note that *any* hyper-parameters should be firstly defined
here to enable yaml and command-line configuration.
"""

_C = CfgNode()

# Logging and saving
_C.LOG = CfgNode()
_C.LOG.SAVE_FREQ = 10
_C.LOG.VIS_FREQ = 1
_C.LOG.VIS_FREQ_STEP = 1000

# Data settings
_C.DATA = CfgNode()
_C.DATA.PATH = ''
_C.DATA.GT_PATH = ''
_C.DATA.MASK_PATH = ''

_C.DATA.NUM_WORKERS = 4
_C.DATA.BATCH_SIZE = 16 #phase2\3: 16 #phase1: 4
_C.DATA.IMG_SIZE = 256
_C.DATA.PART_JITTER_RATE = 0.8

# Training hyper-parameters
_C.TRAINING = CfgNode()
_C.TRAINING.G_LR = 1e-5 #phase1\3: 5e-5 #phase2: 1e-3
_C.TRAINING.D_LR = 1e-5
_C.TRAINING.BETA1 = 0.5
_C.TRAINING.BETA2 = .999
_C.TRAINING.NUM_EPOCHS = 50 # phase1\3: 50 #phase2: 100
_C.TRAINING.LR_DECAY_FACTOR = 5e-2
_C.TRAINING.DOUBLE_D = False
_C.TRAINING.PHASE2_loadpath = ''
_C.TRAINING.PHASE3_loadpath = ''
_C.TRAINING.DIRECT_PHASE3_loadpath = ''

# Loss weights
_C.LOSS = CfgNode()
_C.LOSS.LAMBDA_A = 10.0
_C.LOSS.LAMBDA_B = 10.0
_C.LOSS.LAMBDA_IDT = 0.5
_C.LOSS.LAMBDA_REC = 10
_C.LOSS.LAMBDA_MAKEUP = 100
_C.LOSS.LAMBDA_SKIN = 0.1
_C.LOSS.LAMBDA_EYE = 1.5
_C.LOSS.LAMBDA_LIP = 1.0
_C.LOSS.LAMBDA_MAKEUP_LIP = _C.LOSS.LAMBDA_MAKEUP * _C.LOSS.LAMBDA_LIP
_C.LOSS.LAMBDA_MAKEUP_SKIN = _C.LOSS.LAMBDA_MAKEUP * _C.LOSS.LAMBDA_SKIN
_C.LOSS.LAMBDA_MAKEUP_EYE = _C.LOSS.LAMBDA_MAKEUP * _C.LOSS.LAMBDA_EYE
_C.LOSS.LAMBDA_VGG = 5e-3

### hair
_C.LOSS.LAMBDA_HAIR = 0.15
_C.LOSS.LAMBDA_MAKEUP_HAIR = _C.LOSS.LAMBDA_MAKEUP * _C.LOSS.LAMBDA_HAIR
### background
_C.LOSS.LAMBDA_BACKGROUND = 0.1
_C.LOSS.LAMBDA_MAKEUP_BACKGROUND = _C.LOSS.LAMBDA_MAKEUP * _C.LOSS.LAMBDA_BACKGROUND
# styleloss
_C.LOSS.LAMBDA_STYLE = 0.05
# ds loss
_C.LOSS.LAMBDA_DS = 1.0
# grad loss
_C.LOSS.LAMBDA_GRAD = 1.0

# L2 loss
_C.LOSS.LAMBDA_L2 = 0
_C.LOSS.LAMBDA_PERCEP = 0.1
_C.LOSS.LAMBDA_LPIPS = 1.0
_C.LOSS.LAMBDA_COLOR = -0.001

# Model structure
_C.MODEL = CfgNode()
_C.MODEL.D_TYPE = 'SN'
_C.MODEL.D_REPEAT_NUM = 3
_C.MODEL.D_CONV_DIM = 64
_C.MODEL.G_CONV_DIM = 64
_C.MODEL.NUM_HEAD = 1
_C.MODEL.DOUBLE_E = False
_C.MODEL.USE_FF = False
_C.MODEL.NUM_LAYER_E = 3
_C.MODEL.NUM_LAYER_D = 2
_C.MODEL.WINDOW_SIZE = 16
_C.MODEL.MERGE_MODE = 'conv'
_C.MODEL.STYLECODE_DIM = 64
_C.MODEL.DOMAINS = 4 # lip, face, eye, hair
## [mask_lip, mask_face, mask_eye_left, mask_eye_right, mask_hair]
_C.MODEL.MAX_CONV_DIM = 512
_C.MODEL.MID_DIM = 16
_C.MODEL.RESBLK_NUM = 4
_C.MODEL.USEMASK = True #False
_C.MODEL.RSIM_SIZE = 64 #128
_C.MODEL.USE_DOUBLE_ENCODER = True
# flow
_C.MODEL.FLOW_DIM = 320 #64*5 #512
_C.MODEL.FLOW_BLOCK_NUM = 8

# Preprocessing
_C.PREPROCESS = CfgNode()
_C.PREPROCESS.UP_RATIO = 0.6 / 0.85  # delta_size / face_size
_C.PREPROCESS.DOWN_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.WIDTH_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.LIP_CLASS =     [12, 13]              #
_C.PREPROCESS.FACE_CLASS =    [1, 10, 14, 7, 8]     #
_C.PREPROCESS.EYEBROW_CLASS = [2, 3]                #
_C.PREPROCESS.EYE_CLASS =     [4, 5]                #
_C.PREPROCESS.LANDMARK_POINTS = 68
_C.PREPROCESS.HAIR_CLASS =    [17]                  #

'''
# 0:
# 1: skin
# 2: l brow
# 3: r brow
# 4: l_eye
# 5: r_eye
# 6: eye glasses
# 7: l_ear
# 8: r_ear
# 9: ear ring
# 10: nose
# 11: mouth
# 12: up lip
# 13: down lip
# 14: neck
# 15: neck_l necklace 
# 16: cloth
# 17: hair
# 18: hat 
'''
# Pseudo ground truth
_C.PGT = CfgNode()
_C.PGT.EYE_MARGIN = 12
_C.PGT.LIP_MARGIN = 4
_C.PGT.ANNEALING = True
_C.PGT.SKIN_ALPHA = 0.3
_C.PGT.SKIN_ALPHA_MILESTONES = (0, 12, 24, 50)
_C.PGT.SKIN_ALPHA_VALUES = (0.2, 0.4, 0.3, 0.2)
_C.PGT.EYE_ALPHA = 0.8
_C.PGT.EYE_ALPHA_MILESTONES = (0, 12, 24, 50)
_C.PGT.EYE_ALPHA_VALUES = (0.6, 0.8, 0.6, 0.4)
_C.PGT.LIP_ALPHA = 0.1
_C.PGT.LIP_ALPHA_MILESTONES = (0, 12, 24, 50)
_C.PGT.LIP_ALPHA_VALUES = (0.05, 0.2, 0.1, 0.0)

# NEW
_C.PGT.HAIR_ALPHA = 0.1
_C.PGT.HAIR_ALPHA_MILESTONES = (0, 12, 24, 50)
_C.PGT.HAIR_ALPHA_VALUES = (0.05, 0.2, 0.1, 0.0)


# Postprocessing
_C.POSTPROCESS = CfgNode()
_C.POSTPROCESS.WILL_DENOISE = False

def get_config()->CfgNode:
    return _C
