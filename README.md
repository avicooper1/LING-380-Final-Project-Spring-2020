# LING-380-Final-Project-Spring-2020

## Training the models

To train the models, uncomment the desired model from run_experiment.sh, and then if submitting a job to a Slurm cluster, (like the one maintained by the Yale HPC Center,) use submit_experiment.sh. The file example_lm.py contains the code for loading the SNLI data, defining the model, and training it.

## Testing the models

To test the trained models on BLiMP data, run blimp_final.py and specify the model type you would like to test. Make sure to comment or uncomment the necessary code in blimp_final.py to allow the model to test, given that testing SPINN requires different functions from testing the normal language models. All trained models can be found in the model_checkpoints directory, and their training information (loss, epochs, etc.) can be found in the training_outputs directory.

## Parsing the text

In the blimp_transitions directory, there are some .json files of BLiMP data that has already been entered through a shift-reduce parser. For testing the models, we use these files. If you would like to recreate this parse, setup the Stanford Parser (instructions in setup.txt) then run tree_gen.py, which will parse each sentence in the BLiMP data, binarize the parse tree, and convert it to a shift-reduce parse. You may change the 'principle_A_domain_2' in tools.get_blimp_data('principle_A_domain_2') on line 34 to the name of any other valid BLiMP data file (found here https://github.com/alexwarstadt/blimp/tree/master/data).
