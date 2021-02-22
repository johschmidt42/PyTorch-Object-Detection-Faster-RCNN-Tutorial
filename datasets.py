import torch
import pathlib
from skimage.io import imread
from torchvision.ops import box_convert
from transformations import map_class_to_int
from typing import List, Dict
from transformations import ComposeDouble, ComposeSingle


class ObjectDetectionDataSet(torch.utils.data.Dataset):
    """
    Builds a dataset with images and their respective targets.
    A target is expected to be a pickled file of a dict
    and should contain at least a 'boxes' and a 'labels' key.
    inputs and targets are expected to be a list of pathlib.Path objects.

    In case your labels are strings, you can use mapping (a dict) to int-encode them.
    Returns a dict with the following keys: 'x', 'x_name', 'y', 'y_name'
    """

    def __init__(self,
                 inputs: List[pathlib.Path],
                 targets: List[pathlib.Path],
                 transform: ComposeDouble = None,
                 use_cache: bool = False,
                 convert_to_format: str = None,
                 mapping: Dict = None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.use_cache = use_cache
        self.convert_to_format = convert_to_format
        self.mapping = mapping

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            from multiprocessing import Pool
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = self.read_images(input_ID, target_ID)

        # From RGBA to RGB
        if x.shape[-1] == 4:
            from skimage.color import rgba2rgb
            x = rgba2rgb(x)

        # Read boxes
        try:
            boxes = torch.from_numpy(y['boxes']).to(torch.float32)
        except TypeError:
            boxes = torch.tensor(y['boxes']).to(torch.float32)

        # Read scores
        if 'scores' in y.keys():
            try:
                scores = torch.from_numpy(y['scores']).to(torch.float32)
            except TypeError:
                scores = torch.tensor(y['scores']).to(torch.float32)

        # Label Mapping
        if self.mapping:
            labels = map_class_to_int(y['labels'], mapping=self.mapping)
        else:
            labels = y['labels']

        # Read labels
        try:
            labels = torch.from_numpy(labels).to(torch.int64)
        except TypeError:
            labels = torch.tensor(labels).to(torch.int64)

        # Convert format
        if self.convert_to_format == 'xyxy':
            boxes = box_convert(boxes, in_fmt='xywh', out_fmt='xyxy')  # transforms boxes from xywh to xyxy format
        elif self.convert_to_format == 'xywh':
            boxes = box_convert(boxes, in_fmt='xyxy', out_fmt='xywh')  # transforms boxes from xyxy to xywh format

        # Create target
        target = {'boxes': boxes,
                  'labels': labels}

        if 'scores' in y.keys():
            target['scores'] = scores

        # Preprocessing
        target = {key: value.numpy() for key, value in target.items()}  # all tensors should be converted to np.ndarrays

        if self.transform is not None:
            x, target = self.transform(x, target)  # returns np.ndarrays

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)
        target = {key: torch.from_numpy(value) for key, value in target.items()}

        return {'x': x, 'y': target, 'x_name': self.inputs[index].name, 'y_name': self.targets[index].name}

    @staticmethod
    def read_images(inp, tar):
        return imread(inp), torch.load(tar)


class ObjectDetectionDatasetSingle(torch.utils.data.Dataset):
    """
    Builds a dataset with images.
    inputs is expected to be a list of pathlib.Path objects.

    Returns a dict with the following keys: 'x', 'x_name'
    """

    def __init__(self,
                 inputs: List[pathlib.Path],
                 transform: ComposeSingle = None,
                 use_cache: bool = False,
                 ):
        self.inputs = inputs
        self.transform = transform
        self.use_cache = use_cache

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            from multiprocessing import Pool
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]

            # Load input and target
            x = self.read_images(input_ID)

        # From RGBA to RGB
        if x.shape[-1] == 4:
            from skimage.color import rgba2rgb
            x = rgba2rgb(x)

        # Preprocessing
        if self.transform is not None:
            x = self.transform(x)  # returns a np.ndarray

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)

        return {'x': x, 'x_name': self.inputs[index].name}

    @staticmethod
    def read_images(inp):
        return imread(inp)
