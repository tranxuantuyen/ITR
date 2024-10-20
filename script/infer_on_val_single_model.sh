#!/bin/bash
# SBATCH --gpus=4
# SBATCH --nodes=1
# SBATCH --job-name=mevis
# SBATCH --partition=gpu,gpu-large
# SBATCH --cpus-per-task=36
# SBATCH --mem=120G
# SBATCH --time=3-00:00:00
# SBATCH --qos=batch-short

cd ..
model_file="/home/s222126678/Documents/meccano/project/ITR/output/exp_10_18/model_0054999.pth"
config_file="/home/s222126678/Documents/meccano/project/ITR/output/exp_10_18/config.yaml"
num_gpus=4


rm -rf output_temp/inference

python train_net_itr.py \
--config-file "$config_file" \
--num-gpus $num_gpus --dist-url auto --eval-only \
MODEL.WEIGHTS "$model_file" \
OUTPUT_DIR output_temp DATASETS.TEST '("mevis_val",)' \
SOLVER.IMS_PER_BATCH $num_gpus
python tools/eval_mevis.py --mevis_pred_path output_temp/inference
