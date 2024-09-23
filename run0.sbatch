#!/bin/bash
#
# Usage:
#    sbatch ~/hug/tcr3d/esm-chains.sbatch
#
#SBATCH --partition prod_preempt
#SBATCH --time=4:00:00			# 4:00:00 = 4 hours
#SBATCH --constraint=GPUMEM_80GB	# GPUMODEL_A10# or maybe GPUMEM_24GB
#SBATCH --job-name run0 		# defaults to name of script file
#SBATCH --output=%x-%j.log  		# %x gets the job name; %j gets the slurm job id

##SBATCH --partition legacy		# GPU mem <= 24 Gb, shortish tests
##SBATCH --constraint=GPU_A100_PCI	# GPUMODEL_A10# or maybe GPUMEM_24GB
## SBATCH --ntasks=2
## SBATCH --nodes=1-2
## SBATCH --time=1-12:00:00		# 1 day 12 hours

#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1		# no hyper-threading
#SBATCH --mem=100G			# cpu total mem
#SBATCH --gres=gpu:1			# N gpus
## SBATCH --mem-per-cpu=12G		# (Alt) 4*12 = 48 Gb total
# multi-gpu, opt. B
## SBATCH --gpus-per-task=1
## SBATCH --gpu-bind=single:1		# bind each process to its own GPU
## SBATCH --error=demo-esmfold-slurm-$j.err   # default uses --output
## SBATCH --mail-type=END,FAIL
## SBATCH --nodelist=cipr-gpu01
#   mytut.py is running esm 3B suggests using a GPU w/ 24 Gb mem, so only these 'legacy' nodes ...
## SBATCH --nodelist=cipr-gpu10,cipr-gpu11
#   but even nicer, run:   sinfo --format "%50N %50f" shows me (partial)
#   NODELIST                                           AVAIL_FEATURES                                    
#   cipr-gpu13                                         GPUMODEL_TITAN-RTX,GPUMEM_24GB                    
#   cipr-gpu[01-04]                                    GPUMODEL_TITANV,GPUMEM_12GB                       
#   cipr-gpu[10-11]                                    GPUMODEL_A10,GPUMEM_24GB                          
#   cipr-gpu[12,14]                                    GPUMODEL_2080TI,GPUMEM_11GB                       
#   cipr-gpu15                                         GPUMODEL_1080TI,GPUMEM_11GB                       
#   cipr-gpu[05-06]                                    GPUMODEL_A100-PCIE,GPUMEM_80GB                    
#   cipr-gpu[07-09]                                    GPUMODEL_A100-SXM4,GPUMEM_80GB                    
#   ca-cipr-gpu[01-02],cipr-gpu[16-18]                 GPUMODEL_A6000,GPUMEM_48GB                        
#   cipr-gpu19                                         GPUMODEL_H100-SXM5,GPUMEM_80GB                    
## SBATCH --constaint="IB&Haswell"
## SBATCH --exclude=cipr-gpu03,cipr-gpu04

echo "hostname `hostname`"
nvidia-smi
# we have NOT run .bashrc ...
#source ~/.bashrc
#source ~/bin/conda.sh		# this defaults to ~/miniconda3, which has env esm-pyt2
# BETTER?
eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"


echo "Is /scratch available?"
ls -l /scratch || true
df -h /scratch || true
# cd "$SLURM_SUBMIT_DIR"
cd ~/hug/vidlabel/LLaVA-NeXT
date

conda activate
. ./run0.sh

date
echo "Goodbye from `hostname`"
