#!/usr/bin/bash
# call script for tf1 container
##SBATCH --gres gpu:1
#SBATCH --partition sim-normal
#SBATCH -n 1
#SBATCH -N 1


# cmrr gpu systems
cmrr_dl_host='gpu4dl'

if [[ $(hostname -s) == "aladdinsane" ]]; then
	container_home=${HOME}/workspace/containers
	work_dir=${HOME}/workspace/dltesting/pytorch_tests
	save_dir=${work_dir}/save
	files_dir=${work_dir}/files
	results_dir=${work_dir}/results
elif [[ $(hostname -s) =~ [^$cmrr_dl_host] ]]; then
	container_home=${HOME}/containers
	work_dir=${HOME}/workspace/dltesting/pytorch_tests
	save_dir=${work_dir}/save
	files_dir=${work_dir}/files	
	results_dir=${work_dir}/results
fi

# create directories, if necessary
[ -d ${save_dir} ] || mkdir -p ${save_dir}
[ -d ${files_dir} ] || mkdir -p ${files_dir}
[ -d ${results_dir} ] || mkdir -p ${results_dir} 

# run containerized pytorch workflow
singularity exec --nv -B ${save_dir}:/save \
	-B ${files_dir}:/files \
	-B ${results_dir}:/results \
	${container_home}/pytorch-1.6.0.sif \
    	python ${work_dir}/mnist_pytorch.py
