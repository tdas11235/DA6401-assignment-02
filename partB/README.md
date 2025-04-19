```partB/```: This has all the scripts for part-B of the assignment.
    - ```google_net.py```: Has two functions- ```get_googlenet_shallow()``` (returns GoogLeNet with only unfrozen fc block) and ```get_googlenet()``` (returns GoogLeNet with unfrozen inception5 and fc blocks). We use the latter for the purpose of finetuning.
    - ```trainer.py```: Defines the ```Trainer``` class. Upon instantiating an object, this loads the train and val data. It also has methods for ```train()``` (to train the model), ```train_epoch()``` (to train one epoch), ```evaluate()``` (to evaluate model on validation data).
    - ```fine_tune.py```: Finetunes GoogLeNet and logs the training and validation metrics onto wandb.
    - ```tester.py```: Defines the ```Tester``` class. Upon instantiating an object, this loads the test data. It also has methods for ```evaluate()``` (to evaluate model on validation data), ```collect()``` to collect samples from test set for the $10 \times 3$ grid, ```show_grid()``` for logging the grid to wandb and ```format_caption()``` to set a caption to the images in the grid.
    - ```test.py```: To log test accuracy and $10 \times 3$ grid to wandb.
    - ```models/```: This folder has the best (on validation accuracy) check-point for the finetuned GoogLeNet model (in .pth format).


# Usage Instructions (after dataset is done in root)

For creating a finetuning
```bash
python fine_tune.py
```
For logging test plots and metrics
```bash
python test.py
```