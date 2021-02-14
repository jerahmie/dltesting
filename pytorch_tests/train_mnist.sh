#!/usr/bin/bash
# call script for tf1 container
##SBATCH --gres gpu:1
#SBATCH -n 1
#SBATCH -N 1

container_home=${HOME}/workspace/containers

if [[ $(hostname -s) == "aladdinsane" ]]; then
	work_dir=${HOME}/workspace/dltesting/pytorch_tests
	save_dir=${work_dir}/save
	files_dir=${work_dir}/files
	results_dir=${work_dir}/results
fi

# run containerized pytorch workflow
singularity exec --nv -B ${save_dir}:/save \
	-B ${files_dir}:/files \
	-B ${results_dir}:/results \
	${container_home}/pytorch-1.6.0.sif \
    	python ${work_dir}/mnist_pytorch.py
