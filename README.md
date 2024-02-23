# ERA v2 S5 Assignment
## Purpose
This assignment refactors the S4 assignment notebook (`.ipynb`) in one main notebook file and couple of `.py` files
## Contents
The files are as follows:  
1. **model.py** -This file contains the python classes that inherits _torch.nn.Module_ that defines the Convolution based neural networks. There are 3 different variations of Networks in the file.
- _Input Layer > Convolution layers > Flattening > Fully Connected layers > Output layer_
- _Input Layer > Convolution layers > Global Average Pooling > Fully Connected layers > Output layer_
- _Input Layer > Convolution layers > AntMan layer > Global Average Pooling > Output layer_
3. **utils.py** - This file contains the functions that helps in performing and training and test and report accuracies and losses for each epoch.
4. **S5.ipynb** - This is the main files that orchestrates the flow by utilising the Network classes and the util function to train a CNN for MNIST dataset.
  
## How to navigate
1. Start from the `S5.ipynb` file. This imports the `model.py`, `utils.py`.
2. While in the cell that uses the `Net()` class from `model.py`, go to `model.py` to see definition of the class
3. While in the cell that uses the any class from `utils.py`, go to `utils.py` to see definition of that function
