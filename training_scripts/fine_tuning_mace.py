import warnings
warnings.filterwarnings("ignore")
from mace.cli.run_train import main as mace_run_train_main
import sys
import logging

def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    mace_run_train_main()


train_mace("/home/CRSid/rds/hpc-work/written_assignment_2/config_3_mixed_further_tuning.yaml")
