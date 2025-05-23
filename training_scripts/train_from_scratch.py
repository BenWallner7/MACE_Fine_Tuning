import torch
torch.cuda.empty_cache()

import warnings
warnings.filterwarnings("ignore")
from mace.cli.run_train import main as mace_run_train_main
import sys
import logging

def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    mace_run_train_main()


train_mace("/data/fast-pc-02/CRSid/fine_tuning_assignment/config_files/config_scratch_3.yaml")

