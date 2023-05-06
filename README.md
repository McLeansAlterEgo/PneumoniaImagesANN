# Pneumonia Images Classification using Artificial Neural Networks

The final project and supporting work for the CS404 Artificial Intelligence paper. This project incorporates a neural network from scratch based on **Kinsley, H., & Kukieła, D. (2019). Neural Networks from Scratch in Python.** The implementation can be found in the following file:
 
 - ***PneumoniaImagesANNScratch.ipynb***


## Proposed Network Structure, Methodology and Results Overview

### Network Structure

The purpose of this project was to explore how capable was a properly designed classic **Artificial Neural Network** in image classification compared to Convolutional Neural Networks. Our approach consists of the following:

 - **Input layer** - *2304 Nodes*
 - **1st Hidden layer** - *1024 Nodes*
 - **2nd Hidden layer** - *512 Nodes*
 - **3rd Hidden layer** - *256 Nodes*
 - **Output layer** - *1 Node*

### Image resolution and Activation Functions
The number of nodes within the network is purely dictated by the resolution of the image, we found that a **48x48** image resolution was adequate (with the images being gray scaled), as higher resolutions resulted in longer runtimes without significant gains in accuracy. Moreover, we used the **TanH** activation functions throughout the network except for the output layer, where **Sigmoid** was used.

### Sampling and Training
**Random sampling without replacement** was exclusively used for selecting the data from the original data, then for selecting the batches. While one single batch cannot contain two of the same images, multiple batches can have overlapping images. The model is set to train for exactly **40 Epochs**. We achieved the best results with this configuration. Lastly, a fixed 46 batch size was used.

### Results overview:
   - *Best results* - ***91% Accuracy, 0.31 Loss***
   - *Average results* - ***~83% Accuracy, 0.37 Loss***

## Requirements

In order to run and test this project, the following is needed:
 - Latest version of **Python3**
 - Latest version of **JupyterLab** and **Jupyter Notebook**
 - The images from the dataset that can be found [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
 - Latest version of **Git (Optional)**
 - All packages at the start of the file

## Installation and How to use

### Clone the repository
You can either use the following command in your terminal or just download it directly from GitHub.

```git
git clone https://github.com/EdinZiga/PneumoniaImagesANN
```

Make sure to remember where you clone it as you will need it in the following steps.

### Download and extract the dataset
The dataset that can be found [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Extract the file into the cloned repository.

### Open the project using Jupyter Notebook
Navigate to the repository within your terminal, and open it using Jupyter Notebook.
```terminal
C:\...\PneumoniaImagesAnn>jupyter notebook
```

### Install the necessary packages
Make sure that you have all the necessary packages installed before running to ensure no errors appear.
Packages used can be found below:

```python
import os
import numpy as np
import pydot
import graphviz
import random
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from tqdm import tqdm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
```

### Confirm the directories are correct
Navigate to the load images section. There are 4 directories that have to be set manually. Make sure they align with the location of the downloaded images.
```python
directory = r"D:\PythonProjects\PneumoninaANN\chest_xray\train\NORMAL"
directory = r"D:\PythonProjects\PneumoninaANN\chest_xray\train\PNEUMONIA"
directory = r"D:\PythonProjects\PneumoninaANN\chest_xray\test\NORMAL" 
directory = r"D:\PythonProjects\PneumoninaANN\chest_xray\test\PNEUMONIA"
```

### Run the blocks of code and go do something else
The network from scratch takes a bit to run, depending on your setup, that might be a while. Prepare some reading material. 

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change. The best way to contribute is to run this on your local machine and see if you can get better results than us. Different network tunings are also welcome.


## License
[GNUv3](https://choosealicense.com/licenses/agpl-3.0/) Feel free to use for whatever you wish, give credit where credit is due.
