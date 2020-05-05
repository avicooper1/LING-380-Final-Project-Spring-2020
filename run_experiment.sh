#!/bin/bash

module load miniconda
source ../.bash_profile
conda activate fpenv
nvidia-smi
python example_lm.py --epochs 1 --model SPINN --glove_set 6B