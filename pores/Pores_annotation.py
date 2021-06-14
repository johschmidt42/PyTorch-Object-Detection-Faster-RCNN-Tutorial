# %% From scratch

import pathlib
from visual import Annotator

from utils import get_filenames_of_path

dir_images = pathlib.Path(r"F:\Juan\object_detection\training_data")
image_files = get_filenames_of_path(dir_images / 'input_224x224')

annotator = Annotator(image_ids=image_files)
annotator.napari()
annotator.viewer.window._qt_window.setGeometry(-1280, 0, 1280, 1400)
annotator.viewer.window._qt_window.raise_()  # raise to foreground


# %% Resume annotation work with:

import pathlib
from visual import Annotator

from utils import get_filenames_of_path

dir_images = pathlib.Path(r"F:\Juan\object_detection\training_data")
image_files = get_filenames_of_path(dir_images / 'input_224x224')
dir_annotations = pathlib.Path(r"F:\Juan\object_detection\training_data")
annotation_files = get_filenames_of_path(dir_annotations / 'target_224x224_bb_camilla')

color_mapping = {
    'open': 'red',
    'closed': 'blue'
}

annotator = Annotator(image_ids=image_files, annotation_ids=annotation_files, color_mapping=color_mapping)
annotator.napari()
annotator.viewer.window._qt_window.setGeometry(-1280, 0, 1280, 1400)
annotator.viewer.window._qt_window.raise_()  # raise to foreground

# %% save annotations
save_dir = pathlib.Path(r"C:\Users\johan\Desktop\Johannes")
annotator.export(save_dir)

save_dir = pathlib.Path(r"C:\Users\johan\Desktop\Johannes\Test")
annotator.export_all(save_dir)
