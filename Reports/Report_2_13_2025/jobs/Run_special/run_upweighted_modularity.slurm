#!/bin/bash
#SBATCH -o upweighted_mod_out.out
#SBATCH -e upweighted_mod_error.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jarredk@uidaho.edu
#SBATCH --mem-per-cpu=20000
#SBATCH --cpus-per-task=4
#SBATCH -p nine

cd /mnt/ceph/jarredk/HGRN_repo/Reports/Report_2_13_2025/jobs/Run_special
echo 'Submitting Simulation job'
date
echo '----------------------------------------'

python3 run_upweighted_modularity.py