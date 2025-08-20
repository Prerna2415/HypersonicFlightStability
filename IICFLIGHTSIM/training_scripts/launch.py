import os
import sys

# Add the IICFLIGHTSIM directory to the Python path
iicflightsim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(iicflightsim_dir)

# Now you can import modules from the environments package
from environments.HypersonicEnv import HypersonicEnv

from training_scripts import training

if __name__ == "__main__":
    # Call the main function or execute the training process directly
    # For example, if training.py has a main function:
    # training.main()

    # Or, if training.py just contains the training code, you can execute it here:
    import training
