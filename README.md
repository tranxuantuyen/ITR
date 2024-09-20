# Preparation
Download the [MeViS Dataset](https://github.com/henghuiding/MeViS?tab=readme-ov-file), the backbone weight `model_final_86143f.pkl` and follow [instruction](https://github.com/henghuiding/MeViS/blob/main/INSTALL.md) for installation

# Run experiment
## Train
```
cd script
sbatch train.sh
```
## Validation
```
# change path for the `model_dir`
cd script
sbatch infer_on_val.sh
```
## Testing
```
# change path for the `model_dir`
cd script
sbatch infer_on_test.sh
```

# Note 
To obtain the results on test set, you need to submit the segmentation result to an online sever at [Codalab](https://codalab.lisn.upsaclay.fr/competitions/15094). The waiting time for server results may be a bit long. You can evaluate the model locally on the validation set and then use the best model for test set evaluation. Currently, ITR achieves around 54-55 J&F on the validation set, and 45.3 on the test set. Performance may vary around ±0.5% with different runs.
