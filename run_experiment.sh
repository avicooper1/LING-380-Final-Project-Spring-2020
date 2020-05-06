#!/bin/bash

module load miniconda
source ../.bash_profile
conda activate fpenv
nvidia-smi
python -u example_lm.py --epochs 50 --model SRN --glove_set 6B --device cuda:0 >> SRN_output.txt
#python -u example_lm.py --epochs 50 --model LSTM --glove_set 6B --device cuda:0 >> LSTM_output.txt
#python -u example_lm.py --epochs 50 --model GRU --glove_set 6B --device cuda:0 >> GRU_output.txt
#python -u example_lm.py --epochs 50 --model SPINN --glove_set 6B --device cuda:0 >> SPINN_output.txt