# Train your own object detector with Faster-RCNN & PyTorch

This repository contains all files that were used for the blog tutorial
[**Train your own object detector with Faster-RCNN & PyTorch**](https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70).

If you want to use neptune for your own experiments, add the 'NEPTUNE' env var to your system.
A complete jupyter notebook can be found [here](training_script.ipynb).

The [visual.py](pytorch_faster_rcnn_tutorial/visual.py) script contains the code to visualize a dataset, a list of images, anchor boxes or to create annotations for a dataset.
The provided code for this script was written with napari [0.4.8](https://napari.org/docs/dev/release/release_0_4_8.html) and can be viewed as a hacky solution instead of quality code with good software engineering. 
With napari being actively developed, you can expect changes that might break the code some time in the future.

## Important
- I changed the target structure from a pickled file with suffix `.pt` to a `.json` file.
- I removed the api_key_neptune.py file, because storing credentials like this is considered bad practice!
  Now you have to get your API key from your systems env var `NEPTUNE`.

Changes in the blog post will follow. Stay tuned.


If you cannot start jupyter-lab or jupyter-notebook on Windows because of 
`ImportError: DLL load failed while importing win32api`,
try to run `conda install pywin32` if using the conda package manager.


Installation steps:

- `conda create -n <env_name>`
- `conda activate <env_name>`
- `conda install python=3.8` 
- `git clone https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial.git`
- `cd PyTorch-Object-Detection-Faster-RCNN-Tutorial`
- `pip install .`
- You have to install a pytorch version with `pip` or `conda` that meets the requirements of your hardware. 
  Otherwise the versions for torch etc. specified in [setup.py](setup.py) are installed.
  To install the correct pytorch version for your hardware, check [pytorch.org](https://pytorch.org/).
- [OPTIONAL] To check whether pytorch uses the nvidia gpu, check if `torch.cuda.is_available()` returns `True` in a python shell.
