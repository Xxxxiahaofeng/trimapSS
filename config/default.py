from yacs.config import CfgNode as CN

_C = CN()
_C.OUTDIR = '/media/xiehaofeng/新加卷/things/CamVid/mywork'
_C.N_EPOCH = 50
_C.MAX_ITER = 100000
# First train global decoder for TRAIN_GLOBAL epochs without focus decoder
_C.TRAIN_FOCUS = 0
_C.RESUME = -1
_C.PRETRAINED = ''
# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TRAIN        = '/media/xiehaofeng/新加卷/things/CamVid'
_C.DATASET.VAL          = '/media/xiehaofeng/新加卷/things/CamVid'
_C.DATASET.N_CLASS      = 32
_C.DATASET.BATCHSIZE    = 4
_C.DATASET.IMGSIZE      = (960, 720)
_C.DATASET.AUGMENTATION = []

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.N_CLASS                    = 32
_C.MODEL.BACKBONE                   = 'resnet50'
_C.MODEL.GLOBAL_DECODER             = 'PPM'
_C.MODEL.FOCUS_DECODER              = True
_C.MODEL.FPN                        = False
# The output channel of FPN neck
_C.MODEL.FPN_OUT                    = 1024
# in channel of PPM module
_C.MODEL.PPM_IN                     = 1024
# The output channel of backbone
_C.MODEL.FC_DIM                     = (128, 256, 512, 1024)
# The adaptive pooling scale of Payramid Pooling Module
_C.MODEL.POOL_SCALES                = (1, 2, 3, 6)
# ground truth image size
_C.MODEL.SEGSIZE                    = (512, 512)
# size of erosion and dilation kernal
_C.MODEL.EROSION_DILATION_KERNAL    = 11
# use instance norm or batch norm in unet
_C.MODEL.NORM_LAYER                 = 'instance'
# if use dropout in unet
_C.MODEL.DROPOUT                    = False
_C.MODEL.RETURN_FEATURE             = False

# ---------------------------------------------------------
# LOSS
# ---------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.TYPE            = 'MixLoss'
_C.LOSS.IGNORE_INDEX    = -1
_C.LOSS.LOSS_WEIGHT     = (1, 1)
_C.LOSS.CLASS_WEIGHT    = None
# The threshold probability in OHEM loss, a pixel will be use when probability less than THRESHOLD
_C.LOSS.THRESHOLD       = 0.7
# minimum number of pixel using in OHEM loss
_C.LOSS.MIN_KEPT        = 100000

# ---------------------------------------------------------
# OPTIMIZER
# ---------------------------------------------------------
_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE           = 'SGD'
_C.OPTIMIZER.LR             = 0.0001
_C.OPTIMIZER.BETA1          = 0.9
_C.OPTIMIZER.BETA2          = 0.999
_C.OPTIMIZER.WEIGHT_DECAY   = 1e-5
_C.OPTIMIZER.LR_SCHEDULER   = 'poly'

# ---------------------------------------------------------
# GAN
# ---------------------------------------------------------
_C.GAN = CN()
_C.GAN.G_LR         = 0.00004
_C.GAN.D_LR         = 0.00001
# The threshold of the noise for ground truth label image and score when training discriminator
_C.GAN.THRESHOLD    = 0.1
_C.GAN.LEAKYRELU    = 0.0
# The times of training discriminator in a iteration
_C.GAN.D_ITER       = 1
# the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
_C.GAN.GAN_MODE     = 'vanilla'

# ---------------------------------------------------------
# CONTRAST
# ---------------------------------------------------------
_C.CONTRAST = CN()
_C.CONTRAST.TEMPERATURE = 0.07
_C.CONTRAST.BASE_TEMPERATURE = 0.07
_C.CONTRAST.MAX_SAMPLES = 1024
_C.CONTRAST.MAX_VIEWS = 100
_C.CONTRAST.LOSS_WEIGHT = 0.1
_C.CONTRAST.QUEUE_SIZE = 512

# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
_C.TEST = CN()
_C.TEST.TESTIMAGE   = '/media/xiehaofeng/新加卷/things/EvLab-SSBenchmark/val/38.tif'
_C.TEST.CHECKPOINT  = '/media/xiehaofeng/新加卷/things/evlab-benchmark-segfix/trimap2.0/checkpoints/69epoch.pth'
_C.TEST.PATCHSIZE   = 1024
_C.TEST.OVERLAP     = 0.5
_C.TEST.OUTDIR      = '/media/xiehaofeng/新加卷/things/evlab-benchmark-segfix/trimap2.0/38.tif'