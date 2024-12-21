import os
from argparse import ArgumentParser
from pathlib import Path
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path

from mmyolo.registry import VISUALIZERS
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import get_file_list, show_data_classes

cfg = r'jiyan/yolov8_s_syncbn_fast_8xb16-500e_coco.py'
checkpoint = r'./jiyan/v8_epoch_30.pth'
#
# cfg = r'jiyan/rtmdet_s_syncbn_fast_8xb32-300e_coco.py'
# checkpoint = r'./jiyan/rtm_epoch_40.pth'
#
# cfg = r'jiyan/yolov10_s_syncbn_fast_8xb16-500e_coco.py'
# checkpoint = r'./jiyan/epoch_90.pth'
#
# cfg = r'jiyan/ppyoloe_plus_s_fast_8xb8-80e_coco.py'
# checkpoint = r'./jiyan/pp_epoch_50.pth'
# image_root = './data/coco/images'
imgs = 'F:\\ACCV\\demo\\1.png'#  # G:\\keqing\\mmyolo\\data\\coco\\images\\1ba028ce-IMG_20240121_165437_1.jpg
thr = 0.05
show = False#True #False
output = './output'

if not show:
    path.mkdir_or_exist(output)
model = init_detector(cfg, checkpoint, device='cuda:0')
# init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

# get file list
files, source_type = get_file_list(imgs)


# start detector inference
progress_bar = ProgressBar(len(files))
for file in files:
    result = inference_detector(model, file)
    img = mmcv.imread(file)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    filename = os.path.basename(file)
    out_file = None if show else os.path.join(output, filename)
    progress_bar.update()


    dataset_classes = ('\nreal', '\nreflect',)
    pred_instances = result[0].pred_instances_state[
        result[0].pred_instances_state.scores > thr]
    result[0].pred_instances = result[0].pred_instances_state
    print(result[0].pred_instances)
    visualizer.add_datasample(
        filename,
        img,
        classes=dataset_classes,
        data_sample=result[0],
        draw_gt=False,
        show=show,
        wait_time=0,
        out_file=out_file,
        pred_score_thr=thr)

    img = mmcv.imread(output +'/' + filename)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    dataset_classes = ('bowl', 'apple', 'mouse', 'keyboard', 'banana',
                       'carrot', 'cup', 'orange', 'chair', 'book',)
    pred_instances = result[0].pred_instances_cls[
        result[0].pred_instances_cls.scores > thr]
    result[0].pred_instances = result[0].pred_instances_cls
    print(result[0].pred_instances)
    visualizer.add_datasample(
                filename,
                img,
                classes=dataset_classes,
                data_sample=result[0],
                draw_gt=False,
                show=show,
                wait_time=0,
                out_file=out_file,
                pred_score_thr=thr)


if not show:
    print_log(
        f'\nResults have been saved at {os.path.abspath(output)}')
print(1)
print(1)
print(1)
