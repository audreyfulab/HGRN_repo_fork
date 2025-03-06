#!/bin/bash
#SBATCH -o complex_out.out
#SBATCH -e complex_error.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jarredk@uidaho.edu
#SBATCH --mem-per-cpu=80000
#SBATCH --cpus-per-task=4
#SBATCH -p gpu-8 --gres=gpu:1

cd /mnt/ceph/jarredk/HGRN_repo/Reports/Report_2_13_2025/jobs/Run_special
echo 'Submitting Simulation job'
date
echo '----------------------------------------'

echo '--system specs--'
nvidia-smi
nvcc --version

echo '-----------------------'

source ~/miniconda3/bin/activate

which python

conda activate HGRNpacks


echo 'running job'
python3 run_complex.py



