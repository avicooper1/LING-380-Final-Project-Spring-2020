#!/bin/bash

module load miniconda
source ../.bash_profile
conda activate fpenv
nvidia-smi
python example_lm.py --epochs 50 --model SRN --glove_set 6B