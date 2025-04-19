# Details about partA code

```partA/```: This has all the scripts for part-A of the assignment.
    - ```cnn.py```: Defines the CNN model class.
    - ```trainer.py```: Defines the ```Trainer``` class. Upon instantiating an object, this loads the train and val data. It also has methods for ```train()``` (to train the model), ```train_epoch()``` (to train one epoch), ```evaluate()``` (to evaluate model on validation data).
    - ```tester.py```: Defines the ```Tester``` class. Upon instantiating an object, this loads the test data. It also has methods for ```evaluate()``` (to evaluate model on validation data), ```collect()``` to collect samples from test set for the $10 \times 3$ grid, ```show_grid()``` for logging the grid to wandb and ```format_caption()``` to set a caption to the images in the grid.
    - ```sweep.py```: The script for wandb sweeps
    - ```sweep_finer.py```: The script for top-5 re-runs.
    - ```config.yaml```: The configuration parameters for wandb sweep.
    - ```test.py```: To log test accuracy and $10 \times 3$ grid to wandb.
    - ```mod.py```: modify the run-table to extract the activation_pair column as strings for plotting in parallel coordinates plot.
    - ```models_top_5/```: This folder has the best (on validation accuracy) check-points for the top-5 models (in .pth format).

# Usage Instructions (after dataset is done in root)

For creating a sweep
```bash
python sweep.py
```
For creating a top-5 re-runs
```bash
python sweep_finer.py
```
For logging test plots and metrics
```bash
python test.py
```
For modifying the plots in wandb by inserting activation functions
```bash
python mod.py
```