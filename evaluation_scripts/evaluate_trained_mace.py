from mace.cli.eval_configs import main as mace_eval_configs_main
import sys

def eval_mace(configs, model, output):
    sys.argv = ["program", "--configs", configs, "--model", model, "--output", output, "--default_dtype", "float64"]
    mace_eval_configs_main()

print("Starting evaluation calls...")

base_path = "/rds/user/CRSid/hpc-work/written_assignment_2/"

test_file = "train_validation_test_sets/bulk_and_confined_datasets_training_50/test_500_structures.xyz"

training_file = "train_validation_test_sets/bulk_and_confined_datasets_training_50/train_50_structures.xyz"

model_file = "/home/CRSid/mace/mace/calculators/foundations_models/mace-mpa-0-medium.model"

output_train = "evaluation_output/foundation_model_50_structures_train.xyz"

output_test = "evaluation_output/foundation_model_50_structures_test.xyz"

#evaluate the training set
#eval_mace(configs = base_path+training_file, model = model_file, output = base_path+output_train)

#evaluate the test set
eval_mace(configs = base_path+test_file, model = model_file, output = base_path+output_test)
