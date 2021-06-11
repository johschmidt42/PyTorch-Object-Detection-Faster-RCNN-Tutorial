import pathlib
from utils import get_filenames_of_path
from skimage.io import imread, imsave
from transformations import re_normalize
from skimage.transform import rescale
from experiments.Pores.Pores_utils import div, padding
import os

root_path = r'C:\Users\johan\Desktop\JuanPrada\Clone6 +_- dTAG - Images'
folder_names = ['Rep1 + WT', 'Rep2', 'Rep3', 'Rescue Experiment March2021']

for folder_name in folder_names:
    input_path = pathlib.Path(root_path, folder_name)
    inputs = get_filenames_of_path(input_path)
    print(input_path, len(inputs))
    for img_path in inputs:
        print('reading', img_path.name)
        img = imread(img_path)
        print('scaling')
        img_scaled = rescale(img, scale=2.0, multichannel=True)
        img_scaled = re_normalize(img_scaled)
        new_size = div(img_scaled, window_shape=(256, 256, 3))
        print('padding')
        img_padded = padding(img_scaled, desired_y=new_size[1], desired_x=new_size[0], constant=0)
        new_folder_path = pathlib.Path(root_path) / 'rescaled' / folder_name
        if not new_folder_path.is_dir():
            os.mkdir(new_folder_path)
        print('new_size:', new_size)
        print('saving')
        new_img_path = new_folder_path / img_path.name
        if not new_img_path.is_file():
            imsave(new_img_path, img_padded)




