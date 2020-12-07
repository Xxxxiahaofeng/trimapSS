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
# TEST
# ---------------------------------------------------------
_C.TEST = CN()
_C.TEST.TESTIMAGE = '/media/xiehaofeng/新加卷/things/EvLab-SSBenchmark/val/38.tif'
_C.TEST.CHECKPOINT = '/media/xiehaofeng/新加卷/things/evlab-benchmark-segfix/trimap2.0/checkpoints/69epoch.pth'
_C.TEST.PATCHSIZE = 1024
_C.TEST.OVERLAP = 0.5
_C.TEST.OUTDIR = '/media/xiehaofeng/新加卷/things/evlab-benchmark-segfix/trimap2.0/38.tif'
