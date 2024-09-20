# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_vita_config(cfg):
    cfg.DATASETS.DATASET_RATIO = []

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    # Pseudo Data Use
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)

    # LSJ
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0

    # VITA
    cfg.MODEL.VITA = CN()
    cfg.MODEL.VITA.NHEADS = 8
    cfg.MODEL.VITA.DROPOUT = 0.0
    cfg.MODEL.VITA.DIM_FEEDFORWARD = 2048
    cfg.MODEL.VITA.ENC_LAYERS = 6
    cfg.MODEL.VITA.DEC_LAYERS = 3
    cfg.MODEL.VITA.ENC_WINDOW_SIZE = 0
    cfg.MODEL.VITA.PRE_NORM = False
    cfg.MODEL.VITA.HIDDEN_DIM = 256
    cfg.MODEL.VITA.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.VITA.ENFORCE_INPUT_PROJ = True

    cfg.MODEL.VITA.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.VITA.DEEP_SUPERVISION = True
    cfg.MODEL.VITA.LAST_LAYER_NUM = 3
    cfg.MODEL.VITA.MULTI_CLS_ON = True
    cfg.MODEL.VITA.APPLY_CLS_THRES = 0.01

    cfg.MODEL.VITA.SIM_USE_CLIP = True
    cfg.MODEL.VITA.SIM_WEIGHT = 0.5

    cfg.MODEL.VITA.FREEZE_DETECTOR = False
    cfg.MODEL.VITA.FREEZE_TEXT_ENCODER = True
    cfg.MODEL.VITA.TEST_RUN_CHUNK_SIZE = 18
    cfg.MODEL.VITA.TEST_INTERPOLATE_CHUNK_SIZE = 5

    cfg.MODEL.VITA.TEST_OUTPUT_THRESHOLD = 0.8

def add_itr_config(cfg):
    cfg.ITR = CN()
    cfg.ITR.SPTIO_TEMP_ENCODER_LAYER = 0
    cfg.ITR.WEIGHT_RESUDIAL_PATH = False
    cfg.ITR.WEIGHT_RESUDIAL_IN_RNN = False
    cfg.ITR.FUSE_VISION_TEXT = "add"
