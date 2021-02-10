#!/usr/bin/bash
# call script for tf1 container
##SBATCH --gres gpu:1
#SBATCH -n 1
#SBATCH -N 1

BASE_DIR=/mnt/Data/kaggle/dogs-vs-cats-small
SAVE_DIR=${BASE_DIR}/save
TRAIN_DATA=${BASE_DIR}/train
VALIDATION_DATA=${BASE_DIR}/validation
container_home=${HOME}/workspace/containers
# run containerized tf1 workflow
singularity exec --nv -B $TRAIN_DATA:/train_data -B $VALIDATION_DATA:/validation_data -B $SAVE_DIR:/save_data ${container_home}/tensorflow1.sif \
    python ${HOME}/workspace/tf1_tests/dogsvcats.py
