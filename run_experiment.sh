#!/bin/bash

conda activate fpenv
nvidia-smi
cd LING-380-Final-Project-Spring-2020
python example_lm.py --epochs 1 --model SPINN --glove_set 6B