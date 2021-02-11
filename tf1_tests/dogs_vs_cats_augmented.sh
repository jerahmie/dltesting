#!/usr/bin/bash
# call script for tf1 container
##SBATCH --gres gpu:1
#SBATCH -n 1
#SBATCH -N 1

if [[ $(hostname -s) == 'aladdinsane' ]]; then 
    BASE_DIR=/mnt/Data/kaggle/dogs-vs-cats-small
    container_home=${HOME}/workspace/containers
else
    BASE_DIR=${HOME}/workspace/dltesting/tf1_tests/
fi

SAVE_DIR=${BASE_DIR}/save
TRAIN_DATA=${BASE_DIR}/train
VALIDATION_DATA=${BASE_DIR}/validation

# run containerized tf1 workflow
singularity exec --nv -B $TRAIN_DATA:/train_data \
    -B $VALIDATION_DATA:/validation_data \
    -B $SAVE_DIR:/save_data \
    ${container_home}/tensorflow_1_15_4.sif \
    python ${HOME}/workspace/dltesting/tf1_tests/dogs_vs_cats_augmented.py
