#!/bin/bash
#SBATCH -o complex_out.out
#SBATCH -e complex_error.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jarredk@uidaho.edu
#SBATCH --mem-per-cpu=30000
#SBATCH --cpus-per-task=4
#SBATCH -p eight

cd /mnt/ceph/jarredk/HGRN_repo/Reports/Report_2_13_2025/jobs/Run_special
echo 'Submitting Simulation job'
date
echo '----------------------------------------'

python3 run_complex.py



