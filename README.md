# INT3405E Project
**Dataset**: Modified [MovieLens](https://grouplens.org/datasets/movielens/). You can download the dataset [here](https://drive.google.com/uc?id=1hUqu1mbFeTEfBvl-7fc56fHFfCSzIktD).

## Installation
*Note: You must have Python 3.9+ to install the required dependencies.*

1. Create and activate a virtual environment
```sh
python -m venv venv
./venv/scripts/activate
```
2. Install requirements
```
pip install -r requirements.txt
```

## Run
To train the model with default hyperparameters, run `train.py` script. The script will automatically download and extract the dataset.
```sh
python train.py
```
## Configuration
To see a list of configurable hyperparameters, run the command
```sh
python train.py --help
```
This will output
```
usage: train.py [-h] [--validation-ratio VALIDATION_RATIO] [--shuffle | --no-shuffle]
                [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--learning-rate LEARNING_RATE]   
                [--track | --no-track] [--wandb-project-name WANDB_PROJECT_NAME]

options:
  -h, --help            show this help message and exit
  --validation-ratio VALIDATION_RATIO
                        how much to split train/validation dataset
  --shuffle, --no-shuffle
                        whether to shuffle dataset (default: False)
  --batch-size BATCH_SIZE
                        Batch size for dataloader
  --epochs EPOCHS       Number of epochs to train
  --learning-rate LEARNING_RATE
                        the learning rate of the optimizer
  --track, --no-track   if toggled, this run will be tracked with Weights and Biases
                        (default: False)
  --wandb-project-name WANDB_PROJECT_NAME
                        the wandb's project name
```