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
#SBATCH --output=../output_slurm/$(date +%m-%d_%H-%M)_%j_%x.out
source activate
conda activate vita 
cd ..
python train_net_itr.py \
    --config-file /home/s222126678/Documents/meccano/project/ITR_project/output/exp_08_09_rnn_residual/config.yaml \
    --num-gpus 4 --dist-url auto --eval-only \
    MODEL.WEIGHTS /home/s222126678/Documents/meccano/project/ITR_project/output/exp_08_09_rnn_residual/save/model_0082499.pth \
    OUTPUT_DIR output/54_66 DATASETS.TEST '("mevis_test",)' \
    SOLVER.BASE_LR 0.000025 \
    SOLVER.MAX_ITER 110000 \
    SOLVER.IMS_PER_BATCH 4