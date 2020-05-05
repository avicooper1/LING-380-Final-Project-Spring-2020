#!/bin/bash

module load miniconda
source ../.bash_profile
conda activate fpenv
nvidia-smi
python example_lm.py --epochs 50 --model SRN --glove_set 6B --device cuda:0 >> ../SRN2_output.txt
#python example_lm.py --epochs 50 --model LSTM --glove_set 6B --device cuda:0 >> ../LSTM2_output.txt
#python example_lm.py --epochs 50 --model GRU --glove_set 6B --device cuda:0 >> ../GRU2_output.txt
#python example_lm.py --epochs 50 --model SPINN --glove_set 6B --device cuda:0 >> ../SPINN2_output.txt