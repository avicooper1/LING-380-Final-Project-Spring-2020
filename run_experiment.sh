#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --ntasks=1 --nodes=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=avi.cooper@yale.edu

conda activate fpenv
cd LING-380-Final-Project-Spring-2020
python example_lm.py --epochs 1 --model SPINN --glove_set 6B