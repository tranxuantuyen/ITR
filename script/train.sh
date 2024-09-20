#!/bin/bash
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --job-name=mevis
#SBATCH --partition=gpu,gpu-large
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH --qos=batch-short
#SBATCH --prefer=gpu-a100
source activate
conda activate vita 
cd ..
python train_net_itr.py \
    --config-file configs/itr_swin_tiny.yaml \
    --num-gpus 4 --dist-url auto \
    MODEL.WEIGHTS model_final_86143f.pkl \
    OUTPUT_DIR output/exp_20_09v2 DATASETS.TEST '("mevis_val",)' \
    SOLVER.BASE_LR 0.000025 \
    SOLVER.MAX_ITER 160000 \
    SOLVER.IMS_PER_BATCH 4 \
    TEST.EVAL_PERIOD 100000000 \
    DATALOADER.NUM_WORKERS 26 \
    SOLVER.CHECKPOINT_PERIOD 2500 \
    ITR.SPTIO_TEMP_ENCODER_LAYER 6 \
    ITR.WEIGHT_RESUDIAL_IN_RNN True \
    ITR.WEIGHT_RESUDIAL_PATH False \
    ITR.FUSE_VISION_TEXT 'add'
