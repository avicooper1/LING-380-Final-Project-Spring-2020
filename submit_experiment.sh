#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:p100:1
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=avi.cooper@yale.edu

cd LING-380-Final-Project-Spring-2020
./run_experiment.sh