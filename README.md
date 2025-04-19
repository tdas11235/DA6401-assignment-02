# DA6401 - Assignment 2

Name: Tarak Das \
Roll No.: CH21B108

## Code Structure

1. Dataset extraction pipeline: 
    - ```data_extractor.py```: Extracts the dataset from the zip file downloaded.
    - ```data_splitter.py```: Splits the dataset into train and val partitions.

2. ```partA/```: This has all the scripts for part-A of the assignment.
    - ```cnn.py```: Defines the CNN model class.
    - ```trainer.py```: Defines the ```Trainer``` class. Upon instantiating an object, this loads the train and val data. It also has methods for ```train()``` (to train the model), ```train_epoch()``` (to train one epoch), ```evaluate()``` (to evaluate model on validation data).
    - ```tester.py```: Defines the ```Tester``` class. Upon instantiating an object, this loads the test data. It also has methods for ```evaluate()``` (to evaluate model on validation data), ```collect()``` to collect samples from test set for the $10 \times 3$ grid, ```show_grid()``` for logging the grid to wandb and ```format_caption()``` to set a caption to the images in the grid.
    - ```sweep.py```: The script for wandb sweeps
    - ```sweep_finer.py```: The script for top-5 re-runs.
    - ```config.yaml```: The configuration parameters for wandb sweep.
    - ```test.py```: To log test accuracy and $10 \times 3$ grid to wandb.
    - ```mod.py```: modify the run-table to extract the activation_pair column as strings for plotting in parallel coordinates plot.
    - ```models_top_5/```: This folder has the best (on validation accuracy) check-points for the top-5 models (in .pth format).

3. ```partB/```: This has all the scripts for part-B of the assignment.
    - ```google_net.py```: Has two functions- ```get_googlenet_shallow()``` (returns GoogLeNet with only unfrozen fc block) and ```get_googlenet()``` (returns GoogLeNet with unfrozen inception5 and fc blocks). We use the latter for the purpose of finetuning.
    - ```trainer.py```: Defines the ```Trainer``` class. Upon instantiating an object, this loads the train and val data. It also has methods for ```train()``` (to train the model), ```train_epoch()``` (to train one epoch), ```evaluate()``` (to evaluate model on validation data).
    - ```fine_tune.py```: Finetunes GoogLeNet and logs the training and validation metrics onto wandb.
    - ```tester.py```: Defines the ```Tester``` class. Upon instantiating an object, this loads the test data. It also has methods for ```evaluate()``` (to evaluate model on validation data), ```collect()``` to collect samples from test set for the $10 \times 3$ grid, ```show_grid()``` for logging the grid to wandb and ```format_caption()``` to set a caption to the images in the grid.
    - ```test.py```: To log test accuracy and $10 \times 3$ grid to wandb.
    - ```models/```: This folder has the best (on validation accuracy) check-point for the finetuned GoogLeNet model (in .pth format).


## Setup Instructions

Create and activate the virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

Set up venv using requirements.txt (at the root)
```bash
pip install -r requirements.txt
```

Set up wandb. Run the following command on terminal and paste the API key from wandb
```bash
wandb login
```


## Dataset Setup

1. Download the dataset from [here](https://storage.googleapis.com/wandb_datasets/nature_12K.zip)
2. Run the following commands
```bash
python data_extractor.py
python data_splitter.py
```

## Usage for part-A
For creating a sweep
```bash
python partA/sweep.py
```
For creating a top-5 re-runs
```bash
python partA/sweep_finer.py
```
For logging test plots and metrics
```bash
python partA/test.py
```
For modifying the plots in wandb by inserting activation functions
```bash
python partA/mod.py
```

## Usage for part-B
For creating a finetuning
```bash
python partB/fine_tune.py
```
For logging test plots and metrics
```bash
python partB/test.py
```

## Note
The same instructions are also repeated in part-A and part-B directories too.