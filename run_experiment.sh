#!/bin/bash

module load miniconda
source ../.bash_profile
conda activate fpenv
nvidia-smi
python example_lm.py --epochs 50 --model SRN --glove_set 6B --device cuda:0 > ../SRN_output.txt 2>&1 &
python example_lm.py --epochs 50 --model LSTM --glove_set 6B --device cuda:1 > ../LSTM_output.txt 2>&1 &
python example_lm.py --epochs 50 --model GRU --glove_set 6B --device cuda:2 > ../GRU_output.txt 2>&1 &
python example_lm.py --epochs 50 --model SPINN --glove_set 6B --device cuda:3 > ../SPINN_output.txt 2>&1 &