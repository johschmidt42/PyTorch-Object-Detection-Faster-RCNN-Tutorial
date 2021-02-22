# %% Imports
import pathlib
from utils import get_filenames_of_path
from utils import from_file_to_BoundingBox
from itertools import chain
from metrics.pascal_voc_evaluator import get_pascalvoc_metrics
from metrics.enumerators import MethodAveragePrecision

# %% root directory
root = pathlib.Path(r"C:\Users\johan\Desktop\Johannes\Heads")

# %% input and target files
inputs = get_filenames_of_path(root / 'input')
targets = get_filenames_of_path(root / 'target')

inputs.sort()
targets.sort()

# get the gt_boxes from disk
gt_boxes = [from_file_to_BoundingBox(file_name, groundtruth=True) for file_name in targets]
# reduce list
gt_boxes = list(chain(*gt_boxes))
# TODO: add predictions
pred_boxes = [from_file_to_BoundingBox(file_name, groundtruth=False) for file_name in targets]
pred_boxes = list(chain(*pred_boxes))

output = get_pascalvoc_metrics(gt_boxes=gt_boxes,
                               det_boxes=pred_boxes,
                               iou_threshold=0.5,
                               method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                               generate_table=True)

per_class, mAP = output['per_class'], output['mAP']
head = per_class['head']

# %% another test:Difference between computing the mAP per batch and then taking the mean and computing it directly from all batches
all_gt = []
all_pred = []
all_mAP = []
all_per_class = []
for batch in dataloader_valid:
    x, y, x_name, y_name = batch
    with torch.no_grad():
        task.model.eval()
        preds = task.model(x)

    from itertools import chain
    from utils import from_dict_to_BoundingBox

    gt_boxes = list(
        chain(*[from_dict_to_BoundingBox(target, name=name, groundtruth=True) for target, name in zip(y, x_name)]))
    pred_boxes = list(
        chain(*[from_dict_to_BoundingBox(pred, name=name, groundtruth=False) for pred, name in zip(preds, x_name)]))

    all_gt.append(gt_boxes)
    all_pred.append(pred_boxes)

    from metrics.pascal_voc_evaluator import get_pascalvoc_metrics
    from metrics.enumerators import MethodAveragePrecision
    metric = get_pascalvoc_metrics(gt_boxes=gt_boxes,
                                   det_boxes=pred_boxes,
                                   iou_threshold=0.5,
                                   method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                   generate_table=False)

    per_class, mAP = metric['per_class'], metric['mAP']
    all_per_class.append(per_class)
    all_mAP.append(mAP)

all_tp = [pc[1]['total TP'] for pc in all_per_class]
all_fp = [pc[1]['total FP'] for pc in all_per_class]


all_gt = list(chain(*all_gt))
all_pred = list(chain(*all_pred))

m = get_pascalvoc_metrics(gt_boxes=all_gt,
                               det_boxes=all_pred,
                               iou_threshold=0.5,
                               method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                               generate_table=True)

per_class, mAP = m['per_class'], m['mAP']