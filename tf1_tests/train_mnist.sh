#!/usr/bin/bash
# call script for tf1 container
##SBATCH --gres gpu:1
#SBATCH -n 1
#SBATCH -N 1

container_home=${HOME}/workspace/containers
# run containerized tf1 workflow
singularity exec --nv ${container_home}/tensorflow1.sif \
    python ${HOME}/workspace/tf1_tests/mnist_demo.py
