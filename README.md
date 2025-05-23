# Mace Fine Tuning

## Dataset setup and exploration

`fine_tuning_investigation.ipynb`: initial exploration of dataset and creating the mixed and confined only train, test and validation splits for the respective dataset sizes.

`input.xyz`: original dataset for graphene-water systems, training and testing sets produced from this initial file.

## Config files

`config_fine_tune.yaml`: config file for fine-tuning a 'medium' MACE-MP-0 foundation model.

`config_5_mixed_further_tuning.yaml`: config file for fine-tuning a 'larger' MACE-MP-0 foundation model with higher 'float64' accuracy.

`config_scratch_20.yaml`: config file for training MACE models from scratch, changed the number of epochs for the respective training set sizes.


## Training Scripts

`fine_tuning_mace.py`: script to run fine-tuning using the MACE train executable. This reads in the respective parameters from the config file.

`train_from_scratch.py`: script to run training from scratch using the MACE train executable. This reads in the respective parameters from the config file.

## Evaluation scripts

`evaluate_trained_mace.py`: runs MACE's evaluation function on training and testing sets inputted with model file included, this outputs 'xyz' files with MACE calculated forces and energies and the reference calculations which can be analysed to compare energy and force errors.

## MD runs for nanoconfined graphene

`layer_1_md_run_script.py`: python script for nanoconfined md simulations with equilibration, adapated from 'https://github.com/water-ice-group/graphene-water-protons/blob/main/mlp-based-md/main/run-ase.py.

`continue_layer_1_md_run_script.py`: continue md run further after equilibration and initial production runs.


## Analysis of fine tuning and trajectories

`fine_tuning_analysis.ipynb`: analysis of fine-tuning vs training from scratch vs foundation model in terms of data efficiency, accuracy and computational cost. Correlations for force and energy of MACE models compared to reference calculations, combining into pearson correlation heat-map. Identifying 1, 2 and 3 layers for nano-confined systems to run simulations. Analysing Molecular Dynamics runs to look at normalized density profles for the 3 system sizes.