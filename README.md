This repository contains all files that were used for the blog tutorial
'Train your own object detector with Faster-RCNN & PyTorch'.
You can find the tutorial [here](https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70).

If you want to use neptune for your own experiments, change [api_key_neptune.py](api_key_neptune.py) accordingly.
A complete jupyter notebook can be found [here](training_script.ipynb).

The [visual.py](visual.py) script contains the code to visualize a dataset, a list of images, anchor boxes or to create annotations for a dataset.
The provided code for this script was written around napari 0.4.5 and can be viewed as a hacky solution instead of quality code with good software engineering. 
With napari being actively developed, you can expect changes that might break the code some time in the future. 