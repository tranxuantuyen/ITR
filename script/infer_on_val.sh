#!/bin/bash
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --job-name=mevis
#SBATCH --partition=gpu,gpu-large
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --qos=batch-short

cd ..
source activate
conda activate vita 
model_dir="/home/s222126678/Documents/meccano/project/ITR_project/output/exp_13_09_concat"

for model_file in "$model_dir"/*.pth
do
    rm -rf output_temp_concat/inference
    echo "======= Model name: $(basename "$model_file") ======="
    python train_net_itr.py \
    --config-file /home/s222126678/Documents/meccano/project/ITR_project/output/exp_13_09_concat/config.yaml \
    --num-gpus 4 --dist-url auto --eval-only \
    MODEL.WEIGHTS "$model_file" \
    OUTPUT_DIR output_temp_concat DATASETS.TEST '("mevis_val",)' \
    SOLVER.IMS_PER_BATCH 4
    python tools/eval_mevis.py --mevis_pred_path output_temp_concat/inference
done
