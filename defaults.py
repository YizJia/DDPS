from yacs.config import CfgNode as CN

_C = CN()

# -------------------------------------------------------- #
#                           Input                          #
# -------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.DATASET = "CUHK-SYSU"
_C.INPUT.DATA_ROOT = "data/CUHK-SYSU"

# Size of the smallest side of the image
_C.INPUT.MIN_SIZE = 900
# Maximum size of the side of the image
_C.INPUT.MAX_SIZE = 1500

# TODO: support aspect ratio grouping
# Whether to use aspect ratio grouping for saving GPU memory
# _C.INPUT.ASPECT_RATIO_GROUPING_TRAIN = False

# Number of images per batch
_C.INPUT.BATCH_SIZE_TRAIN = 5
_C.INPUT.BATCH_SIZE_TEST = 1

# Number of data loading threads
_C.INPUT.NUM_WORKERS_TRAIN = 5
_C.INPUT.NUM_WORKERS_TEST = 1

# -------------------------------------------------------- #
#                          Solver                          #
# -------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 20

# Learning rate settings
_C.SOLVER.BASE_LR = 0.003

# TODO: add config option WARMUP_EPOCHS
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
# _C.SOLVER.WARMUP_EPOCHS = 1

# The epoch milestones to decrease the learning rate by GAMMA
_C.SOLVER.LR_DECAY_MILESTONES = [16]
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.SGD_MOMENTUM = 0.9

# Loss weight of RPN regression
_C.SOLVER.LW_RPN_REG = 1
# Loss weight of RPN classification
_C.SOLVER.LW_RPN_CLS = 1
# Loss weight of proposal regression
_C.SOLVER.LW_PROPOSAL_REG = 10
# Loss weight of proposal classification
_C.SOLVER.LW_PROPOSAL_CLS = 1
# Loss weight of box regression
_C.SOLVER.LW_BOX_REG = 1
# Loss weight of box classification
_C.SOLVER.LW_BOX_CLS = 1
# Loss weight of box OIM (i.e. Online Instance Matching)
_C.SOLVER.LW_BOX_REID = 1

# Set to negative value to disable gradient clipping
_C.SOLVER.CLIP_GRADIENTS = 10.0

# -------------------------------------------------------- #
#                            RPN                           #
# -------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.RPN = CN()
# Anchor sizes
_C.MODEL.ANCHOR_SIZES = ((32, 64, 128, 256, 512),)
# Anchor aspect_ratios base setting
_C.MODEL.ANCHOR_ASPECT_RATIOS = ((0.5, 1.0, 2.0),)
# NMS threshold used on RoIs
_C.MODEL.RPN.NMS_THRESH = 0.7
# Number of anchors per image used to train RPN
_C.MODEL.RPN.BATCH_SIZE_TRAIN = 256
# Target fraction of foreground examples per RPN minibatch
_C.MODEL.RPN.POS_FRAC_TRAIN = 0.5
# Overlap threshold for an anchor to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.RPN.POS_THRESH_TRAIN = 0.7
# Overlap threshold for an anchor to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.RPN.NEG_THRESH_TRAIN = 0.3
# Number of top scoring RPN RoIs to keep before applying NMS
_C.MODEL.RPN.PRE_NMS_TOPN_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPN_TEST = 6000
# Number of top scoring RPN RoIs to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOPN_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPN_TEST = 300
# rpn score threshold
_C.MODEL.RPN.SCORE_THRESH = 0.0

# -------------------------------------------------------- #
#                         RoI head                         #
# -------------------------------------------------------- #
_C.MODEL.ROI_HEAD = CN()
# Whether to use bn neck (i.e. batch normalization after linear)
_C.MODEL.ROI_HEAD.BN_NECK = True
# Number of RoIs per image used to train RoI head
_C.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN = 128
# Target fraction of foreground examples per RoI minibatch
_C.MODEL.ROI_HEAD.POS_FRAC_TRAIN = 0.5
# Overlap threshold for an RoI to be considered foreground (if >= POS_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.POS_THRESH_TRAIN = 0.5
# Overlap threshold for an RoI to be considered background (if < NEG_THRESH_TRAIN)
_C.MODEL.ROI_HEAD.NEG_THRESH_TRAIN = 0.5
# Minimum score threshold
_C.MODEL.ROI_HEAD.SCORE_THRESH_TEST = 0.5
# NMS threshold used on boxes
_C.MODEL.ROI_HEAD.NMS_THRESH_TEST = 0.4
# Maximum number of detected objects
_C.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST = 300

# -------------------------------------------------------- #
#                           Loss                           #
# -------------------------------------------------------- #
_C.MODEL.LOSS = CN()
# Size of the lookup table in OIM
_C.MODEL.LOSS.LUT_SIZE = 5532
# Size of the circular queue in OIM
_C.MODEL.LOSS.CQ_SIZE = 5000
_C.MODEL.LOSS.OIM_MOMENTUM = 0.5
_C.MODEL.LOSS.OIM_SCALAR = 30.0

# -------------------------------------------------------- #
#                   Baseline Single model                  #
# -------------------------------------------------------- #
_C.BASELINE = CN()
_C.BASELINE.PROJECT = True
_C.BASELINE.TYPE = "baseline"
_C.BASELINE.BACKBONE_NAME = "resnet50"
_C.BASELINE.BACKBONE_PRETRAIN = True
# _C.BASELINE.BACKBONE_EXTRA_BLOCKS = None
_C.BASELINE.BACKBONE_NECK = False
_C.BASELINE.TRAINABLE_LABYER = 2
_C.BASELINE.RETURNED_LAYERS = ['0', '1', '2', '3', 'pool']
_C.BASELINE.BACKBONE_OUTCHANNEL = 1024
_C.BASELINE.RES5HEAD_OUTCHANNEL = [1024, 2048]
_C.BASELINE.FASTERHEAD_INCHANNEL = 2048
_C.BASELINE.BBOXREG_INCHANNEL = 2048
_C.BASELINE.NAE_INCHANNEL = [1024, 2048]
_C.BASELINE.FROZENBN2d = True
_C.BASELINE.FREEZE = True

# -------------------------------------------------------- #
#                          Distiller                       #
# -------------------------------------------------------- #
_C.DISTILLER = CN()
_C.DISTILLER.PROJECT = False
_C.DISTILLER.TYPE = "NONE"
_C.DISTILLER.INIT_STUDENT = False

_C.DISTILLER.TEACHER = CN()
_C.DISTILLER.TEACHER.PRETRAINED = True
_C.DISTILLER.TEACHER.BACKBONE_NAME = "resnet50"
_C.DISTILLER.TEACHER.BACKBONE_PRETRAIN = True
_C.DISTILLER.TEACHER.BACKBONE_NECK = False
_C.DISTILLER.TEACHER.TRAINABLE_LABYER = 5
_C.DISTILLER.TEACHER.RETURNED_LAYERS = ['0', '1', '2', '3', 'pool']
_C.DISTILLER.TEACHER.BACKBONE_OUTCHANNEL = 1024
_C.DISTILLER.TEACHER.RES5HEAD_OUTCHANNEL = [1024,2048]
_C.DISTILLER.TEACHER.FASTERHEAD_INCHANNEL = 2048
_C.DISTILLER.TEACHER.BBOXREG_INCHANNEL = 2048
_C.DISTILLER.TEACHER.NAE_INCHANNEL = [1024,2048]
_C.DISTILLER.TEACHER.FREEZE = True
_C.DISTILLER.TEACHER.FROZENBN2d = True
_C.DISTILLER.TEACHER.CKPT = "exp_cuhk/FGD/pre_epoch_19.pth"

_C.DISTILLER.STUDENT = CN()
_C.DISTILLER.STUDENT.BACKBONE_NAME = "resnet18"
_C.DISTILLER.STUDENT.BACKBONE_PRETRAIN = True
_C.DISTILLER.STUDENT.BACKBONE_NECK = False
_C.DISTILLER.TEACHER.TRAINABLE_LABYER = 2
_C.DISTILLER.STUDENT.RETURNED_LAYERS = ['0', '1', '2', '3', 'pool']
_C.DISTILLER.STUDENT.BACKBONE_OUTCHANNEL = 256
_C.DISTILLER.STUDENT.RES5HEAD_OUTCHANNEL = [256,512]
_C.DISTILLER.STUDENT.FASTERHEAD_INCHANNEL = 512
_C.DISTILLER.STUDENT.BBOXREG_INCHANNEL = 512
_C.DISTILLER.STUDENT.NAE_INCHANNEL = [256,512]
_C.DISTILLER.STUDENT.FREEZE = True
_C.DISTILLER.STUDENT.FROZENBN2d = True

# FGD(Focal and Global Knowledge Distillation) -- Detection
_C.FGD = CN()
_C.FGD.FIND_UNSUED_PARAMETERS = True
_C.FGD.TEMP = 0.5
_C.FGD.ALPHA_FGD = 0.00005
_C.FGD.BATE_FGD = 0.000025
_C.FGD.GAMMA_FGD = 0.00005
_C.FGD.LAMBDA_FGD = 0.0000005
_C.FGD.DISTILLER_METHOD_NUMBER = 4
_C.FGD.TEACHER_CHANNELS = [256, 256, 256, 256]
_C.FGD.STUDENT_CHANNELS = [256, 256, 256, 256]
_C.FGD.LW_FGD = [0.1, 1, 1, 0.1]
_C.FGD.WARMUP = 20.0

# PKD (Pearson Knowledge Distillation) -- Detection
_C.PKD = CN()
_C.PKD.WITHFOCAL = True
_C.PKD.WEIGHT = 6.0
_C.PKD.WHOLE_FGD = True

# ReviewKD -- feature distillation
_C.REVIEWKD = CN()
_C.REVIEWKD.WEIGHT = 1.2

# detection cls KD -- Detection/Classification Head
_C.DETKD = CN()
_C.DETKD.T = 4.0
_C.DETKD.WARMUP = 20.0
_C.DETKD.FROM_T = False

# DKD(Decoupled Knowledge Distillation) -- Classification
_C.DKD = CN()
_C.DKD.ALPHA = 1.0
_C.DKD.BETA = 0.5
_C.DKD.T = 4.0
_C.DKD.FROM_T = False

# reid KD -- Re_ID/OIM
_C.REIDKD = CN()
_C.REIDKD.T = 4.0
_C.REIDKD.WARMUP = 20.0
_C.REIDKD.FROM_T = False

# GraphRelation KD -- reid branch
_C.GRAPHRELA = CN()
_C.GRAPHRELA.FROM_T = False
_C.GRAPHRELA.WEIGHT = 0.01
_C.GRAPHRELA.WARMUP = 20.0
_C.GRAPHRELA.NMSTHRES = 0.7

# -------------------------------------------------------- #
#                        Evaluation                        #
# -------------------------------------------------------- #
# The period to evaluate the model during training
_C.EVAL_PERIOD = 1
# Evaluation with GT boxes to verify the upper bound of person search performance
_C.EVAL_USE_GT = False
# Fast evaluation with cached features
_C.EVAL_USE_CACHE = False
# Evaluation with Context Bipartite Graph Matching (CBGM) algorithm
_C.EVAL_USE_CBGM = False

# -------------------------------------------------------- #
#                           Miscs                          #
# -------------------------------------------------------- #
# Save a checkpoint after every this number of epochs
_C.CKPT_PERIOD = 1
# The period (in terms of iterations) to display training losses
_C.DISP_PERIOD = 10
# Whether to use tensorboard for visualization
_C.TF_BOARD = True
# The device loading the model
_C.DEVICE = "cuda"
# Set seed to negative to fully randomize everything
_C.SEED = 1
# Directory where output files are written
_C.OUTPUT_DIR = "./output"


def get_default_cfg():
    """
    Get a copy of the default config.
    """
    ## Q1: clone()是什么层次的拷贝 Q2: _C的初始化是什么时候执行的？
    return _C.clone()
