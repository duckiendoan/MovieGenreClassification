# INT3405E Project - Group 25

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
usage: train.py [-h] [--seed SEED] [--validation-ratio VALIDATION_RATIO]
                [--shuffle | --no-shuffle] [--batch-size BATCH_SIZE] [--epochs EPOCHS]        
                [--learning-rate LEARNING_RATE] [--asl | --no-asl] [--model MODEL]
                [--language-model LANGUAGE_MODEL] [--track | --no-track]
                [--wandb-project-name WANDB_PROJECT_NAME] [--save-model | --no-save-model]    
                [--model-path MODEL_PATH]

options:
  -h, --help            show this help message and exit
  --seed SEED           seed of the experiment
  --validation-ratio VALIDATION_RATIO
                        how much to split train/validation dataset
  --shuffle, --no-shuffle
                        whether to shuffle dataset (default: False)
  --batch-size BATCH_SIZE
                        Batch size for dataloader
  --epochs EPOCHS       Number of epochs to train
  --learning-rate LEARNING_RATE
                        the learning rate of the optimizer
  --asl, --no-asl       whether to use Asymmetrical Loss (default: True)
  --model MODEL         The model to use from models module
  --language-model LANGUAGE_MODEL
                        Language model used for movie title processing from HuggingFace       
                        transformers
  --track, --no-track   if toggled, this run will be tracked with Weights and Biases
                        (default: False)
  --wandb-project-name WANDB_PROJECT_NAME
                        the wandb's project name
  --save-model, --no-save-model
                        whether to save model after training (default: False)
  --model-path MODEL_PATH
                        the path to load model from
```

## Reproducing results in the report
We ran the following variants during the project:

- ResNet50 + DistilBERT + ASL
```sh
python train.py --asl --model JointModel --epochs 40
```
- ResNet50 + DistilBERT + BCE
```sh
python train.py --no-asl --model JointModel --epochs 40
```
- ResNet50 + ASL
```sh
python train.py --asl --model ImageOnlymodel --epochs 40
```
- ResNet50 + BCE
```sh
python train.py --no-asl --model ImageOnlymodel --epochs 40
```